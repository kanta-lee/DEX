import numpy as np
import torch
from cvxopt import solvers
from cvxopt.base import matrix


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return np.array(sol['x']).flatten()


class PositionCLF():
    def __init__(self, net: torch.nn.Module, device: torch.device):
        self.net = net
        self.device = device
        self.x_dim = 3
        self.u_dim = 3

        self._traj_states = {}

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
        if key not in self._traj_states:
            if key == 'NeedlePick':
                self._init_spiral_state(env)
            elif key == 'GauzeRetrieve':
                self._init_wipe_state(env)
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

        epsilon = 5.0
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
    def __init__(self, net: torch.nn.Module, device: torch.device):
        self.net = net
        self.device = device
        self.x_dim = 6
        self.u_dim = 4

        self._traj_states = {}

    def _init_spiral_state(self, env):
        _spiral_horizon = 60
        _spiral_turns = 2.0
        _spiral_turns_ori = 1.0

        goal = np.asarray(env.goal, dtype=np.float32)
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        yaw_start = np.asarray(env._get_robot_state(0)[[5]], dtype=np.float32)
        steps = max(_spiral_horizon, 1)

        center_xy = 0.5 * (start[:2] + goal[:2])

        r_start = np.linalg.norm(start[:2] - center_xy)+0.05
        r_goal = np.linalg.norm(goal[:2] - center_xy)+0.05

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
        robot_state = np.asarray(env._get_robot_state(0), dtype=np.float32)
        psm_pos = robot_state[0:3]
        pos_threshold = getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0)
        ori_threshold = 0.05

        if (np.linalg.norm(psm_pos - goal) < pos_threshold
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

        epsilon = 20.0
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
