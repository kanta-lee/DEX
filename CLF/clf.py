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
        self._spiral_states = {}
        self._theta_rate = 0.35
        self._radius_decay = 0.98
        self._spiral_horizon = 80
        self._spiral_turns = 3.0

    def _init_spiral_state(self, env):
        key = id(env)
        goal = np.asarray(env.goal, dtype=np.float32)
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        print('start position:', start)
        print('goal position:', goal)

        steps = max(self._spiral_horizon, 1)

        center_xy = 0.5 * (start[:2] + goal[:2])

        r_start = np.linalg.norm(start[:2] - center_xy)
        r_goal = np.linalg.norm(goal[:2] - center_xy)

        theta_start = np.arctan2(start[1] - center_xy[1], start[0] - center_xy[0])
        theta_goal = np.arctan2(goal[1] - center_xy[1], goal[0] - center_xy[0])
        theta_end = theta_goal + 2 * np.pi * self._spiral_turns

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

        self._spiral_states[key] = {
            'goal': goal,
            'start': start,
            'traj': traj,
        }

    def _get_spiral_reference(self, env):
        key = id(env)
        goal = np.asarray(env.goal, dtype=np.float32)
        start = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        needs_reset = key not in self._spiral_states
        if not needs_reset:
            state = self._spiral_states[key]
            needs_reset = not np.allclose(state['goal'], goal) or not np.allclose(state['start'], start)

        if needs_reset:
            self._init_spiral_state(env)

        state = self._spiral_states[key]
        psm_pos = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        dists = np.linalg.norm(state['traj'] - psm_pos, axis=1)
        idx = min(int(np.argmin(dists)) + 1, len(state['traj']) - 1)
        p_ref = state['traj'][idx]

        return p_ref

    @torch.no_grad()
    def needle_pick_spiral(self, u, env):
        # Only engage CLF after the needle is grasped.
        if not hasattr(env, "_activated") or env._activated < 0:
            return u
        # Stop CLF after the needle is close to the goal.
        goal = np.asarray(env.goal, dtype=np.float32)
        state = np.asarray(env._get_robot_state(0)[:3], dtype=np.float32)
        if np.linalg.norm(state - goal) < getattr(env, 'DISTANCE_THRESHOLD', 0.005) * getattr(env, 'SCALING', 1.0):
            return u

        p_ref = self._get_spiral_reference(env)

        psm_pos = env._get_robot_state(0)[:3]
        psm_pos_t = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0).to(self.device)

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

        epsilon = 1.0
        G = LgV.to(self.device)
        h = (-epsilon * V - LfV).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        try:
            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        except Exception:
            modified_u = u

        return modified_u
