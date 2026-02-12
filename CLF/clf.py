import numpy as np
import torch
from cvxopt import solvers
from cvxopt.base import matrix

import pybullet as p
from surrol.utils.pybullet_utils import (
    get_link_pose
)
from torch.fx.experimental.unification.unification_tools import first


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return np.array(sol['x']).flatten()


class PositionCLF():
    # we learn the dynamics of the position of the end-effector only, and use CLF to track a position trajectory
    def __init__(self, net: torch.nn.Module, device: torch.device):
        self.net = net
        self.device = device
        self.x_dim = 3
        self.u_dim = 3

        self._traj_states = {}
        self.traj_idx = 0

    def reset_trajectory(self, env_name=None):
        """Reset trajectory state for a new episode."""
        if env_name is not None:
            if env_name in self._traj_states:
                del self._traj_states[env_name]
        else:
            self._traj_states = {}
        self.traj_idx = 0

    def _init_reach_state(self, env):
        """
        Initialize trajectory from environment reach points with linear interpolation.
        Position-only version (no yaw tracking).
        """
        _interp_points = 2  # Number of interpolation points between waypoints
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(env.goal, dtype=np.float32) + np.array([0.0, 0.0, 0.05])
        
        traj = []
        positions = np.linspace(start, goal, _interp_points)
        
        for k in range(_interp_points):
            traj.append([positions[k][0], positions[k][1], positions[k][2]])
        
        # Ensure the final point reaches the goal
        traj = np.asarray(traj, dtype=np.float32)
        
        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj
        }

    def _init_waypoint_state(self, env):
        """
        Initialize trajectory from environment waypoints with linear interpolation.
        Position-only version (no yaw tracking).
        
        Each waypoint has format: [x, y, z, yaw, gripper_state]
        - gripper_state > 0: open
        - gripper_state < 0: closed
        """
        _interp_points = 2  # Number of interpolation points between waypoints
        
        # Get waypoints from environment
        waypoints = env._waypoints.copy()
        if waypoints is None or len(waypoints) < 2:
            raise ValueError("Environment must have at least 2 waypoints defined")
        
        # Filter out None waypoints
        valid_waypoints = [wp for wp in waypoints if wp is not None]
        if len(valid_waypoints) < 2:
            raise ValueError("Environment must have at least 2 valid waypoints")
        
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(valid_waypoints[-1][:3], dtype=np.float32)
        
        traj = []
        gripper_states = []
        
        # First segment: from current position to first waypoint
        wp0 = valid_waypoints[0]
        pos_start = start
        pos_end = np.asarray(wp0[:3], dtype=np.float32)
        gripper = wp0[4]
        
        positions = np.linspace(pos_start, pos_end, _interp_points)
        for k in range(_interp_points):
            traj.append([positions[k][0], positions[k][1], positions[k][2]])
            gripper_states.append(gripper)
        
        # Subsequent segments: between consecutive waypoints
        for i in range(len(valid_waypoints) - 1):
            wp_curr = valid_waypoints[i]
            wp_next = valid_waypoints[i + 1]
            
            pos_start = np.asarray(wp_curr[:3], dtype=np.float32)
            pos_end = np.asarray(wp_next[:3], dtype=np.float32)
            gripper = wp_next[4]
            
            positions = np.linspace(pos_start, pos_end, _interp_points)
            for k in range(_interp_points):
                traj.append([positions[k][0], positions[k][1], positions[k][2]])
                gripper_states.append(gripper)
        
        # Ensure the final point reaches the goal
        final_wp = valid_waypoints[-1]
        traj.append([final_wp[0], final_wp[1], final_wp[2]])
        gripper_states.append(final_wp[4])
        
        traj = np.asarray(traj, dtype=np.float32)
        gripper_states = np.asarray(gripper_states, dtype=np.float32)
        
        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
            'gripper_states': gripper_states,
            'traj_type': 'waypoints',
        }

    def _init_spiral_state(self, env):
        _spiral_horizon = 30
        _spiral_turns = 3.0

        goal = np.asarray(env.goal, dtype=np.float32)
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        steps = max(_spiral_horizon, 1)

        center_xy = 0.5 * (start[:2] + goal[:2])

        r_start = np.linalg.norm(start[:2] - center_xy)+0.05
        r_goal = np.linalg.norm(goal[:2] - center_xy)-0.01

        theta_start = np.arctan2(start[1] - center_xy[1], start[0] - center_xy[0])
        theta_goal = np.arctan2(goal[1] - center_xy[1], goal[0] - center_xy[0])
        theta_end = theta_goal + 2 * np.pi * _spiral_turns

        radii = np.linspace(r_start, r_goal, steps)
        thetas = np.linspace(theta_start, theta_end, steps)
        zs = np.linspace(start[2], goal[2], steps)

        traj = []
        for k in range(steps):
            x_ref = center_xy[0] + radii[k] * np.cos(thetas[k])
            y_ref = center_xy[1] + radii[k] * np.sin(thetas[k])
            traj.append([x_ref, y_ref, zs[k]])

        # Ensure the final point reaches the goal.
        traj.append(goal.copy())
        traj = np.asarray(traj, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def _init_wipe_state(self, env):
        _wipe_horizon = 40
        _amp = 0.5
        _num_zigzags = 3

        goal = np.asarray(env.goal, dtype=np.float32)
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        steps = max(_wipe_horizon, 2)

        # Build a zigzag path between start and goal in the XY plane with a gentle lateral offset.
        delta_xy = goal[:2] - start[:2]
        norm_xy = np.linalg.norm(delta_xy*2)
        if norm_xy < 1e-6:
            # Default to an arbitrary direction if start and goal overlap in XY.
            delta_xy = np.array([1.0, 0.0], dtype=np.float32)
            norm_xy = 1.0
        dir_xy = delta_xy / norm_xy
        # Perpendicular direction to create the zigzag offsets.
        perp_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float32)

        # Offset amplitude: scale with environment but keep it bounded by the path length.
        scale = getattr(env, 'SCALING', 1.0)
        max_amp = _amp * norm_xy
        amp = min(0.01 * scale, max_amp if max_amp > 1e-6 else 0.01 * scale)

        t_vals = np.linspace(0.0, 1.0, steps)
        traj = []
        for t in t_vals:
            base_xy = start[:2] + t * delta_xy
            offset_mag = amp * np.sin(2 * np.pi * _num_zigzags * t)
            xy = base_xy + offset_mag * perp_xy
            z = start[2] + t * (goal[2] - start[2])
            traj.append([xy[0], xy[1], z])

        # Ensure the final point reaches the goal.
        traj.append(goal.copy())
        traj = np.asarray(traj, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def _get_reference(self, env):
        key = env.__class__.__name__
        # Environments that use waypoint-based trajectories
        waypoint_envs = ('NeedlePickSphere', 'NeedlePickCylinder', 'GauzeRetrieveCylinder', 'GauzeRetrieveSphere')
        reach_envs = ('NeedleReachSphere', 'NeedleReachPlate')
        if key not in self._traj_states:
            if key == 'NeedlePick':
                self._init_spiral_state(env)
            elif key == 'GauzeRetrieve':
                self._init_wipe_state(env)
            elif key in waypoint_envs:
                self._init_waypoint_state(env)
            elif key in reach_envs:
                self._init_reach_state(env)
            else:
                raise ValueError("Unsupported environment for CLF, such as no trajectory defined for this env.")
            self.traj_idx = 0
        state = self._traj_states[key]

        # Stop fetch ref trajectory after the needle is close to the goal.
        goal = np.asarray(env.goal, dtype=np.float32)
        psm_pos = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        if np.linalg.norm(psm_pos - goal) < getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0):
            p_ref = state['traj'][-1]
        # Step forward if close to the current reference point.
        elif (np.linalg.norm(psm_pos - state['traj'][self.traj_idx]) <
              getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0)):
            self.traj_idx = min(self.traj_idx + 1, len(state['traj']) - 1)
            p_ref = state['traj'][self.traj_idx]
        else:
            p_ref = state['traj'][self.traj_idx]

        return p_ref

    def get_current_gripper_state(self, env):
        """Get the gripper state for the current trajectory index."""
        key = env.__class__.__name__
        if key not in self._traj_states:
            return None
        state = self._traj_states[key]
        if 'gripper_states' not in state:
            return None
        
        idx = min(self.traj_idx, len(state['gripper_states']) - 1)
        return state['gripper_states'][idx]

    @torch.no_grad()
    def traj_tracking_waypoints(self, u, env):
        """
        Trajectory tracking for waypoint-based trajectories.
        Returns modified action, reference point, and gripper state.
        Position-only version (no yaw tracking).
        """
        p_ref = self._get_reference(env)
        gripper_state = self.get_current_gripper_state(env)
        psm_pos = env._get_robot_state(0)[:3]

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        psm_pos_t = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)

        with torch.enable_grad():
            psm_pos_t.requires_grad_(True)
            V = 0.5 * torch.sum((psm_pos_t - p_ref_t) ** 2)
            V.backward()
            grad_V = psm_pos_t.grad.detach()

        psm_pos_t.requires_grad_(False)

        net_out = self.net(psm_pos_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T

        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u
        #add zero yaw angle to the modified_u to make it 4D Note: yaw angle will be set to 0 in the env.step
        zero_tensor = torch.zeros(1, 1, device=modified_u.device)
        zero_tensor = zero_tensor.expand(modified_u.shape[0], 1)
        modified_u = torch.cat([modified_u, zero_tensor], dim=-1)
        return modified_u, p_ref, gripper_state

    @torch.no_grad()
    def traj_tracking(self, u, env):
        # Only engage CLF after the needle is grasped.
        if not hasattr(env, "_activated") or env._activated < 0:
            return u, None

        p_ref = self._get_reference(env)
        psm_pos = env._get_robot_state(0)[:3]

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        psm_pos_t = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)

        with torch.enable_grad():
            psm_pos_t.requires_grad_(True)
            V = 0.5 * torch.sum((psm_pos_t - p_ref_t) ** 2)
            V.backward()
            grad_V = psm_pos_t.grad.detach()

        psm_pos_t.requires_grad_(False)

        net_out = self.net(psm_pos_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T

        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u, p_ref


class CLF():
    # we learn the dynamics of the position and orientation of the end-effector, and use CLF to track a trajectory.
    def __init__(self, net: torch.nn.Module, device: torch.device, traj_type: str = 'line'):
        self.net = net
        self.device = device
        self.x_dim = 6
        self.u_dim = 4
        self.traj_type = traj_type  # 'line', 'circle', or 'triangle'

        self._traj_states = {}

    def reset_trajectory(self, env_name=None):
        """
        Reset trajectory state for a new episode.
        Should be called at the start of each episode for environments
        where waypoints depend on object positions (e.g., NeedlePick).
        
        Args:
            env_name: If provided, only reset for this specific environment.
                      If None, reset all trajectory states.
        """
        if env_name is not None:
            if env_name in self._traj_states:
                del self._traj_states[env_name]
        else:
            self._traj_states = {}
        self.traj_idx = 0

    def _init_trajectory_state(self, env, traj_type='line'):
        """
        Initialize trajectory state with selectable trajectory types.
        
        Args:
            env: The environment
            traj_type: 'line' (直线), 'circle' (转圈), or 'triangle' (三角形)
        """
        _horizon = 50  # Number of waypoints
        
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(env.goal, dtype=np.float32)
        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)
        
        traj = []
        
        if traj_type == 'line':
            # 直线轨迹: Linear interpolation from start to goal
            _horizon = 2
            positions = np.linspace(start, goal, _horizon)
            yaws = np.linspace(yaw_start, yaw_start, _horizon)  # Keep yaw constant
            
            for k in range(_horizon):
                traj.append([positions[k][0], positions[k][1], positions[k][2], 
                            np.remainder(yaws[k] + np.pi, 2 * np.pi) - np.pi])
                            
        elif traj_type == 'circle':
            # 转圈轨迹: Circular closed path returning to start
            # goal is the same as start for circle trajectory
            
            # Circle center offset from start, with start at pi/4 position
            circle_radius = 0.12  # Radius of the circle
            theta_start = np.pi / 2  # 起点位置在 pi/4 (45度)
            center_xy = start[:2] - circle_radius * np.array([np.cos(theta_start), np.sin(theta_start)], dtype=np.float32)
            z_center = start[2]  # Keep z constant at start height
    
            theta_end = theta_start - 2 * np.pi  # Full circle back to start
            goal = np.array([center_xy[0] + circle_radius * np.cos(theta_end), center_xy[1] + circle_radius * np.sin(theta_end), start[2]])
            thetas = np.linspace(theta_start, theta_end, _horizon)
            yaws = np.linspace(yaw_start, yaw_start, _horizon)
            
            for k in range(_horizon):
                x_ref = center_xy[0] + circle_radius * np.cos(thetas[k])
                y_ref = center_xy[1] + circle_radius * np.sin(thetas[k])
                traj.append([x_ref, y_ref, start[2], 
                            np.remainder(yaws[k] + np.pi, 2 * np.pi) - np.pi])
            
            # Store circle parameters for deviation calculation
            circle_params = {
                'center_xy': center_xy,
                'radius': circle_radius,
                'z_center': z_center,
            }
                            
        elif traj_type == 'triangle':
            # 三角形轨迹: Equilateral triangular closed path returning to start
            # goal is the same as start for triangle trajectory
            goal = start.copy()
            _horizon = 6
            # Create an equilateral triangle with start as one vertex
            triangle_size = 0.1  # Side length of the equilateral triangle
            
            # Three vertices of the equilateral triangle (start is vertex 0)
            # For equilateral triangle: height = side * sqrt(3) / 2
            triangle_height = triangle_size * np.sqrt(3) / 2
            vertex1 = start + np.array([triangle_size, 0, 0], dtype=np.float32)
            vertex2 = start + np.array([triangle_size / 2, -triangle_height, 0], dtype=np.float32)
            
            # Store triangle vertices for deviation calculation (closed triangle: start->v1->v2->start)
            triangle_vertices = [start.copy(), vertex1.copy(), vertex2.copy(), start.copy()]
            
            # Segments (three edges of the triangle)
            seg1_horizon = _horizon // 3
            seg2_horizon = _horizon // 3
            seg3_horizon = _horizon - seg1_horizon - seg2_horizon
            
            # Segment 1: start -> vertex1
            pos1 = np.linspace(start, vertex1, seg1_horizon)
            yaw1 = np.linspace(yaw_start, yaw_start, seg1_horizon)
            for k in range(seg1_horizon):
                traj.append([pos1[k][0], pos1[k][1], pos1[k][2],
                            np.remainder(yaw1[k] + np.pi, 2 * np.pi) - np.pi])
            
            # Segment 2: vertex1 -> vertex2
            pos2 = np.linspace(vertex1, vertex2, seg2_horizon)
            yaw2 = np.linspace(yaw_start, yaw_start, seg2_horizon)
            for k in range(seg2_horizon):
                traj.append([pos2[k][0], pos2[k][1], pos2[k][2],
                            np.remainder(yaw2[k] + np.pi, 2 * np.pi) - np.pi])
            
            # Segment 3: vertex2 -> start (return to starting point)
            pos3 = np.linspace(vertex2, start, seg3_horizon)
            yaw3 = np.linspace(yaw_start, yaw_start, seg3_horizon)
            for k in range(seg3_horizon):
                traj.append([pos3[k][0], pos3[k][1], pos3[k][2],
                            np.remainder(yaw3[k] + np.pi, 2 * np.pi) - np.pi])
        else:
            raise ValueError(f"Unsupported trajectory type: {traj_type}. Use 'line', 'circle', or 'triangle'.")
        
        # Ensure the final point reaches the goal
        traj.append([goal[0], goal[1], goal[2], np.remainder(yaw_start + np.pi, 2 * np.pi) - np.pi])
        traj = np.asarray(traj, dtype=np.float32)
        
        # Build the state dictionary
        state_dict = {
            'goal': goal,
            'start': start,
            'traj': traj,
            'traj_type': traj_type,
        }
        
        # Add trajectory-specific parameters for deviation calculation
        if traj_type == 'circle':
            state_dict['circle_params'] = circle_params
        elif traj_type == 'triangle':
            state_dict['triangle_vertices'] = triangle_vertices
        
        self._traj_states[env.__class__.__name__] = state_dict

    def _init_linear_state(self, env):
        """Initialize a simple linear trajectory from start to goal with periodic gripper opening/closing."""
        _line_horizon = 40  # Number of waypoints for linear trajectory
        _gripper_frequency = 20.0  # Number of open/close cycles per trajectory (frequency)
        _gripper_open_value = 0.5  # Gripper open state (> 0)
        _gripper_close_value = -0.5  # Gripper closed state (< 0)

        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(env.goal, dtype=np.float32) + np.array([0.035, 0.0, -0.06], dtype=np.float32)
        goal_2 = start + np.array([0.045, 0.0, -0.06], dtype=np.float32)

        # Linear interpolation from start to goal
        pos = np.linspace(start, goal, _line_horizon)
        pos2 = np.linspace(goal, goal_2, _line_horizon)

        # Keep yaw constant (or interpolate if needed)
        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)
        yaw_goal = yaw_start  # Keep same yaw, or set to desired value
        yaws = np.linspace(yaw_start, yaw_goal, _line_horizon)

        traj = []
        gripper_states = []
        total_steps = _line_horizon * 2  # Total steps for both stages
        
        #stage 1 : move above the goal
        for k in range(_line_horizon):
            traj.append([pos[k][0], pos[k][1], pos[k][2], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])
            # Periodic gripper: use sine wave to alternate between open and close
            t = k / total_steps  # Normalized time [0, 1]
            sine_val = np.sin(2 * np.pi * _gripper_frequency * t)
            # Map sine [-1, 1] to [close, open]
            gripper_state = _gripper_close_value if sine_val < 0 else _gripper_open_value
            gripper_states.append(gripper_state)
        
        #stage 2 : move back
        for k in range(_line_horizon):
            traj.append([pos2[k][0], pos2[k][1], pos2[k][2], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])
            # Continue periodic gripper for stage 2
            t = (_line_horizon + k) / total_steps  # Normalized time [0, 1]
            sine_val = np.sin(2 * np.pi * _gripper_frequency * t)
            gripper_state = _gripper_close_value if sine_val < 0 else _gripper_open_value
            gripper_states.append(gripper_state)
        
        traj = np.asarray(traj, dtype=np.float32)
        gripper_states = np.asarray(gripper_states, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
            'gripper_states': gripper_states,
        }

    def _init_linear_state_cbf(self, env):
        """Initialize a simple linear trajectory from start to goal."""
        _line_horizon = 50  # Number of waypoints for linear trajectory

        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        # goal = np.asarray(env.goal, dtype=np.float32) + np.array([0.0, 0.0, -0.06], dtype=np.float32)
        goal = start + np.array([0.0, -0.3, 0.00], dtype=np.float32)
        goal_2 = start + np.array([0.06, 0.0, -0.06], dtype=np.float32)

        # Linear interpolation from start to goal
        pos = np.linspace(start, goal, _line_horizon)
        pos2 = np.linspace(goal, goal_2, _line_horizon)

        # Keep yaw constant (or interpolate if needed)
        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)
        yaw_goal = yaw_start  # Keep same yaw, or set to desired value
        yaws = np.linspace(yaw_start, yaw_goal, _line_horizon)

        traj = []
        #stage 1 : move above the goal
        for k in range(_line_horizon):
            traj.append([pos[k][0], pos[k][1], pos[k][2], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])
        
        # #stage 2 : move back
        # for k in range(_line_horizon):
        #     traj.append([pos2[k][0], pos2[k][1], pos2[k][2], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])
        traj = np.asarray(traj, dtype=np.float32)


        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def _init_spiral_state(self, env):
        _line_horizon = 10
        # initial and middle z bias
        z_bias_1 = 0.075
        z_bias_2 = 0.045
        _needle_radius = 0.07
        _spiral_horizon = 40
        _spiral_turns = 0.75

        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(env.goal, dtype=np.float32)

        # first stage: move the needle close to the target
        pos_start = start
        # theta_end = 0.25 * np.pi
        theta_end = np.pi / 2
        center = goal + np.array([_needle_radius, 0.0, z_bias_1])
        pos_end = np.array([center[0] + _needle_radius * np.cos(theta_end),
                            center[1] + _needle_radius * np.sin(theta_end),
                            center[2]])

        pos = np.linspace(pos_start, pos_end, _line_horizon)

        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)
        # yaw_end = 0.25 * np.pi
        yaw_end = theta_end

        yaws = np.linspace(yaw_start, yaw_end, _line_horizon)

        traj = []
        for k in range(_line_horizon):
            traj.append([pos[k][0], pos[k][1], pos[k][2], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])

        # second stage: rotate the needle to pass through the target
        theta_start_2 = theta_end
        theta_end_2 = theta_start_2 + 2 * np.pi * _spiral_turns
        thetas = np.linspace(theta_start_2, theta_end_2, _spiral_horizon)

        yaw_start_2 = yaw_end
        yaw_end_2 = yaw_start_2 + 2 * np.pi * _spiral_turns
        yaws = np.linspace(yaw_start_2, yaw_end_2, _spiral_horizon)
        # first half z
        zs_1 = np.linspace(goal[2]+z_bias_1, goal[2]+z_bias_2, _spiral_horizon//2)
        zs_2 = np.linspace(goal[2]+z_bias_2, goal[2]+z_bias_1, _spiral_horizon//2)
        zs = np.concatenate(([zs_1, zs_2]))
        for k in range(_spiral_horizon):
            x_ref = center[0] + _needle_radius * np.cos(thetas[k])
            y_ref = center[1] + _needle_radius * np.sin(thetas[k])
            traj.append([x_ref, y_ref, zs[k], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])

        traj = np.asarray(traj, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def _init_sine_state(self, env):
        """
        Initialize a sinusoidal trajectory for NeedlePickWoundCLF.
        Stage 1: Linear approach from start to a point above the wound.
        Stage 2: Sinusoidal path — the needle moves forward while oscillating
                 perpendicular to the main direction of travel.
        """
        _line_horizon = 10
        _sine_horizon = 40
        _sine_cycles = 2.0        # number of full sine wave cycles
        _sine_amplitude = 0.08    # amplitude of the sine oscillation (meters)
        z_bias_1 = 0.03

        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        goal = np.asarray(env.goal, dtype=np.float32)
        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)

        # --- Stage 1: Linear approach to a point above the wound ---
        approach_target = goal + np.array([0.0, 0.0, z_bias_1], dtype=np.float32)
        pos_stage1 = np.linspace(start, approach_target, _line_horizon)
        yaws_stage1 = np.linspace(yaw_start, yaw_start, _line_horizon)

        traj = []
        for k in range(_line_horizon):
            traj.append([pos_stage1[k][0], pos_stage1[k][1], pos_stage1[k][2],
                         np.remainder(yaws_stage1[k] + np.pi, 2 * np.pi) - np.pi])

        # --- Stage 2: Sinusoidal oscillation path ---
        # Main travel direction: from approach_target moving forward (along x)
        travel_vec = np.array([0.2, 0.0, 0.0], dtype=np.float32)  # travel distance
        sine_end = approach_target + travel_vec

        # Build orthogonal direction for oscillation (perpendicular in XY plane)
        travel_dir = travel_vec / (np.linalg.norm(travel_vec) + 1e-8)
        perp_dir = np.array([-travel_dir[1], travel_dir[0], 0.0], dtype=np.float32)

        t_values = np.linspace(0, 1, _sine_horizon)
        yaws_stage2 = np.linspace(yaw_start, yaw_start, _sine_horizon)

        for k in range(_sine_horizon):
            t = t_values[k]
            # Position along the main travel direction
            base_pos = approach_target + t * travel_vec
            # Sinusoidal oscillation perpendicular to travel direction
            sine_offset = _sine_amplitude * np.sin(2 * np.pi * _sine_cycles * t)
            pos = base_pos + sine_offset * perp_dir
            traj.append([pos[0], pos[1], pos[2],
                         np.remainder(yaws_stage2[k] + np.pi, 2 * np.pi) - np.pi])

        # Ensure final point
        traj.append([sine_end[0], sine_end[1], sine_end[2],
                     np.remainder(yaw_start + np.pi, 2 * np.pi) - np.pi])
        traj = np.asarray(traj, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
            'traj_type': 'sine',
        }

    def _init_waypoint_state(self, env):
        """
        Initialize trajectory from environment waypoints with linear interpolation.
        Used for NeedlePick task to follow waypoints for grasping and lifting.
        
        Each waypoint has format: [x, y, z, yaw, gripper_state]
        - gripper_state > 0: open
        - gripper_state < 0: closed
        """
        _interp_points = 2  # Number of interpolation points between waypoints
        
        # Get waypoints from environment
        waypoints = env._waypoints.copy()
        if waypoints is None or len(waypoints) < 2:
            raise ValueError("Environment must have at least 2 waypoints defined")
        
        # Filter out None waypoints
        valid_waypoints = [wp for wp in waypoints if wp is not None]
        if len(valid_waypoints) < 2:
            raise ValueError("Environment must have at least 2 valid waypoints")
        
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        yaw_start = np.asarray(env._get_robot_state(0)[5], dtype=np.float32)
        goal = np.asarray(valid_waypoints[-1][:3], dtype=np.float32)
        
        traj = []
        gripper_states = []  # Track gripper state for each trajectory point
        waypoint_indices = [0]  # Track which trajectory index corresponds to each waypoint
        
        # First segment: from current position to first waypoint
        wp0 = valid_waypoints[0]
        pos_start = start
        pos_end = np.asarray(wp0[:3], dtype=np.float32)
        yaw_end = wp0[3]
        gripper = wp0[4]
        
        positions = np.linspace(pos_start, pos_end, _interp_points)
        yaws = np.linspace(yaw_start, yaw_end, _interp_points)
        
        for k in range(_interp_points):
            traj.append([positions[k][0], positions[k][1], positions[k][2],
                        np.remainder(yaws[k] + np.pi, 2 * np.pi) - np.pi])
            gripper_states.append(gripper)
        
        waypoint_indices.append(len(traj))
        
        # Subsequent segments: between consecutive waypoints
        for i in range(len(valid_waypoints) - 1):
            wp_curr = valid_waypoints[i]
            wp_next = valid_waypoints[i + 1]
            
            pos_start = np.asarray(wp_curr[:3], dtype=np.float32)
            pos_end = np.asarray(wp_next[:3], dtype=np.float32)
            yaw_start_seg = wp_curr[3]
            yaw_end_seg = wp_next[3]
            gripper = wp_next[4]  # Use the gripper state of the target waypoint
            
            positions = np.linspace(pos_start, pos_end, _interp_points)
            yaws = np.linspace(yaw_start_seg, yaw_end_seg, _interp_points)
            
            for k in range(_interp_points):
                traj.append([positions[k][0], positions[k][1], positions[k][2],
                            np.remainder(yaws[k] + np.pi, 2 * np.pi) - np.pi])
                gripper_states.append(gripper)
            
            waypoint_indices.append(len(traj))
        
        # Ensure the final point reaches the goal
        final_wp = valid_waypoints[-1]
        traj.append([final_wp[0], final_wp[1], final_wp[2],
                    np.remainder(final_wp[3] + np.pi, 2 * np.pi) - np.pi])
        gripper_states.append(final_wp[4])
        
        traj = np.asarray(traj, dtype=np.float32)
        gripper_states = np.asarray(gripper_states, dtype=np.float32)
        
        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
            'gripper_states': gripper_states,
            'waypoint_indices': waypoint_indices,
            'traj_type': 'waypoints',
        }

    def yaw_difference(self, yaw1, yaw2):
        diff = yaw1 - yaw2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return np.abs(diff)

    def _get_reference(self, env):
        key = env.__class__.__name__
        if key not in self._traj_states:
            if key == 'NeedlePickWoundCLF':
                if self.traj_type == 'sine':
                    self._init_sine_state(env)
                else:
                    self._init_spiral_state(env)
            elif key == 'NeedlePickLungCLF':
                self._init_linear_state(env)
            elif key == 'NeedlePickTrajectoryCLF':
                self._init_trajectory_state(env, traj_type=self.traj_type)
            elif key == 'NeedlePickTrajectoryCBF':
                self._init_trajectory_state(env, traj_type=self.traj_type)
            elif key == 'NeedlePick' or key == 'NeedlePickSphere' or key == 'NeedlePickCylinder':
                self._init_waypoint_state(env)
            elif key == 'PegTransferPlate' or key == 'PegTransferSphere':
                self._init_waypoint_state(env)
            else:
                raise ValueError("Unsupported environment for CLF, such as no trajectory defined for this env.")
            self.traj_idx = 0
        state = self._traj_states[key]

        # Stop fetch ref trajectory after the needle is close to the goal.
        robot_state = np.asarray(env._get_robot_state(0), dtype=np.float32)
        psm_pos = robot_state[0:3]
        pos_threshold = getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0)
        ori_threshold = 0.05

        # Only allow jumping to final point after completing most of the trajectory (80%)
        # This prevents jumping to end at start for closed trajectories (circle, triangle)
        traj_progress_threshold = 0.8
        has_sufficient_progress = self.traj_idx >= len(state['traj']) * traj_progress_threshold

        if (has_sufficient_progress
                and np.linalg.norm(psm_pos - state['traj'][-1][0:3]) < pos_threshold
                and self.yaw_difference(robot_state[5], state['traj'][-1][3]) < ori_threshold):
            p_ref = state['traj'][-1]
        # Step forward if close to the current reference point.
        elif (np.linalg.norm(psm_pos - state['traj'][self.traj_idx][0:3]) < pos_threshold
              and self.yaw_difference(robot_state[5], state['traj'][self.traj_idx][3]) < ori_threshold):
            self.traj_idx = min(self.traj_idx + 1, len(state['traj']) - 1)
            p_ref = state['traj'][self.traj_idx]
            # if the robot yaw is close to the boundary, there will be a jump in angle difference
            if robot_state[5]-p_ref[3]>np.pi:
                p_ref[3] += 2*np.pi
            elif p_ref[3]-robot_state[5]>np.pi:
                p_ref[3] -= 2*np.pi
        else:
            p_ref = state['traj'][self.traj_idx]

        return p_ref

    def get_current_gripper_state(self, env):
        """
        Get the gripper state for the current trajectory index.
        Returns None if gripper_states is not available (non-waypoint trajectories).
        """
        key = env.__class__.__name__
        if key not in self._traj_states:
            return None
        state = self._traj_states[key]
        if 'gripper_states' not in state:
            return None
        
        idx = min(self.traj_idx, len(state['gripper_states']) - 1)
        return state['gripper_states'][idx]

    @torch.no_grad()
    def traj_tracking_waypoints(self, u, env):
        """
        Trajectory tracking for waypoint-based trajectories (e.g., NeedlePick).
        Returns modified action, reference point, and gripper state.
        Does not check for needle grasping activation.
        """
        p_ref = self._get_reference(env)
        gripper_state = self.get_current_gripper_state(env)
        psm_pos_ori = env._get_robot_state(0)[:6]

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        psm_pos_ori_t = torch.from_numpy(psm_pos_ori).float().unsqueeze(0).to(self.device)

        with torch.enable_grad():
            psm_pos_ori_t.requires_grad_(True)
            V = 0.5 * torch.sum((torch.concat((psm_pos_ori_t[:, :3], psm_pos_ori_t[:, [5]]), dim=1) - p_ref_t) ** 2)
            V.backward()
            grad_V = psm_pos_ori_t.grad.detach()

        psm_pos_ori_t.requires_grad_(False)

        net_out = self.net(psm_pos_ori_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T
        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u, p_ref, gripper_state

    @torch.no_grad()
    def traj_tracking_linear(self, u, env):
        """Linear trajectory tracking without needle grasping check."""
        p_ref = self._get_reference(env)
        psm_pos_ori = env._get_robot_state(0)[:6]
        gripper_state = self.get_current_gripper_state(env)  # Get gripper state if available

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        psm_pos_ori_t = torch.from_numpy(psm_pos_ori).float().unsqueeze(0).to(self.device)

        with torch.enable_grad():
            psm_pos_ori_t.requires_grad_(True)
            V = 0.5 * torch.sum((torch.concat((psm_pos_ori_t[:, :3], psm_pos_ori_t[:, [5]]), dim=1) - p_ref_t) ** 2)
            V.backward()
            grad_V = psm_pos_ori_t.grad.detach()

        psm_pos_ori_t.requires_grad_(False)

        net_out = self.net(psm_pos_ori_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T
        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u, p_ref, gripper_state

    @torch.no_grad()
    def traj_tracking(self, u, env):
        # Only engage CLF after the needle is grasped.
        if not hasattr(env, "_activated") or env._activated < 0:
            return u, None

        p_ref = self._get_reference(env)
        psm_pos_ori = env._get_robot_state(0)[:6]

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        psm_pos_ori_t = torch.from_numpy(psm_pos_ori).float().unsqueeze(0).to(self.device)

        with torch.enable_grad():
            psm_pos_ori_t.requires_grad_(True)
            V = 0.5 * torch.sum((torch.concat((psm_pos_ori_t[:, :3], psm_pos_ori_t[:, [5]]), dim=1) - p_ref_t) ** 2)
            V.backward()
            grad_V = psm_pos_ori_t.grad.detach()

        psm_pos_ori_t.requires_grad_(False)

        net_out = self.net(psm_pos_ori_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T

        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u, p_ref


class ObjCLF():
    # We learn the dynamics of the position, orientation of the end-effector,
    # and the positions of the two ends of the needle.
    # Use CLF to track a trajectory of one end of the needle.
    def __init__(self, net: torch.nn.Module, device: torch.device):
        self.net = net
        self.device = device
        self.x_dim = 6
        self.u_dim = 4

        self._traj_states = {}

    def get_left_needle_pos(self, env):
        left_90_pos, left_90_orn = get_link_pose(env.obj_id, 6)
        left_90_pos = np.array(left_90_pos)
        left_90_orn = np.array(p.getEulerFromQuaternion(left_90_orn))
        return left_90_pos, left_90_orn

    def _init_spiral_state(self, env):
        _spiral_horizon = 60
        _spiral_turns = 1.0
        _spiral_turns_ori = 1.0

        goal = np.asarray(env.goal, dtype=np.float32)

        needle_pos, needle_ori = self.get_left_needle_pos(env)
        start = np.asarray(needle_pos, dtype=np.float32)
        yaw_start = np.asarray(needle_ori[[2]], dtype=np.float32)
        steps = max(_spiral_horizon, 1)

        center_xy = 0.5 * (start[:2] + goal[:2])

        r_start = np.linalg.norm(start[:2] - center_xy)+0.05
        r_goal = np.linalg.norm(goal[:2] - center_xy)-0.01

        theta_start = np.arctan2(start[1] - center_xy[1], start[0] - center_xy[0])
        theta_goal = np.arctan2(goal[1] - center_xy[1], goal[0] - center_xy[0])
        theta_end = theta_goal + 2 * np.pi * _spiral_turns

        radii = np.linspace(r_start, r_goal, steps)
        thetas = np.linspace(theta_start, theta_end, steps)
        zs = np.linspace(start[2], goal[2], steps)

        yaw_end = yaw_start + 2 * np.pi * _spiral_turns_ori
        yaws = np.linspace(yaw_start[0], yaw_end[0], steps)

        traj = []
        for k in range(steps):
            x_ref = center_xy[0] + radii[k] * np.cos(thetas[k])
            y_ref = center_xy[1] + radii[k] * np.sin(thetas[k])
            traj.append([x_ref, y_ref, zs[k], np.remainder(yaws[k]+np.pi, 2 * np.pi)-np.pi])

        # Ensure the final point reaches the goal.
        traj.append([goal[0], goal[1], goal[2], (np.remainder(yaw_end+np.pi, 2 * np.pi)-np.pi)[0]])
        traj = np.asarray(traj, dtype=np.float32)

        self._traj_states[env.__class__.__name__] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def yaw_difference(self, yaw1, yaw2):
        diff = yaw1 - yaw2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return np.abs(diff)

    def _get_reference(self, env):
        key = env.__class__.__name__
        if key not in self._traj_states:
            if key == 'NeedlePick':
                self._init_spiral_state(env)
            else:
                raise ValueError("Unsupported environment for CLF, such as no trajectory defined for this env.")
            self.traj_idx = 0
        state = self._traj_states[key]

        # Stop fetch ref trajectory after the needle is close to the goal.
        goal = np.asarray(env.goal, dtype=np.float32)
        needle_pos, needle_ori = self.get_left_needle_pos(env)
        needle_left_pos = np.asarray(needle_pos, dtype=np.float32)
        pos_threshold = getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0)
        ori_threshold = 0.05

        if (np.linalg.norm(needle_left_pos - goal) < pos_threshold
            and self.yaw_difference(needle_ori[2], state['traj'][-1][3]) < ori_threshold):
            p_ref = state['traj'][-1]
        # Step forward if close to the current reference point.
        elif (np.linalg.norm(needle_left_pos - state['traj'][self.traj_idx][0:3]) < pos_threshold
              and self.yaw_difference(needle_ori[2], state['traj'][self.traj_idx][3]) < ori_threshold):
            self.traj_idx = min(self.traj_idx + 1, len(state['traj']) - 1)
            p_ref = state['traj'][self.traj_idx]
            # if the robot yaw is close to the boundary, there will be a jump in angle difference
            print(needle_ori[2], p_ref[3])
            if needle_ori[2]-p_ref[3]>np.pi:
                p_ref[3] += 2*np.pi
            elif p_ref[3]-needle_ori[2]>np.pi:
                p_ref[3] -= 2*np.pi
        else:
            p_ref = state['traj'][self.traj_idx]

        return p_ref


    @torch.no_grad()
    def traj_tracking(self, u, env):
        # Only engage CLF after the needle is grasped.
        if not hasattr(env, "_activated") or env._activated < 0:
            return u, None

        p_ref = self._get_reference(env)
        needle_left_pos, needle_left_ori = self.get_left_needle_pos(env)

        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)
        needle_left_pos_t = torch.from_numpy(needle_left_pos).float().unsqueeze(0).to(self.device)
        needle_left_ori_t = torch.from_numpy(needle_left_ori).float().unsqueeze(0).to(self.device)

        with ((torch.enable_grad())):
            needle_left_pos_t.requires_grad_(True)
            needle_left_ori_t.requires_grad_(True)
            V = 0.5 * torch.sum((needle_left_pos_t - p_ref_t[:, 0:3]) ** 2
                                )+0.5 * torch.sum((needle_left_ori_t[:, [2]] - p_ref_t[:, [3]]) ** 2)
            # V = 0.5 * torch.sum((needle_left_pos_t - p_ref_t[:, 0:3]) ** 2)
            V.backward()
            grad_V = torch.concat((needle_left_pos_t.grad.detach(), needle_left_ori_t.grad.detach()), dim=1)
            # grad_V = torch.concat((needle_left_pos_t.grad.detach(), torch.zeros((1,3), device=self.device)), dim=1)

        needle_left_pos_t.requires_grad_(False)
        needle_left_ori_t.requires_grad_(False)

        obs = np.concatenate((needle_left_pos, needle_left_ori))
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        net_out = self.net(obs_t)
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))

        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T

        epsilon = 15.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = torch.zeros(self.u_dim)

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u, p_ref
 