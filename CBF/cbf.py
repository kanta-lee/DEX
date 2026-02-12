import numpy as np
import torch
from cvxopt import solvers
from cvxopt.base import matrix


def qp_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return np.array(sol['x']).flatten()


class CBF():
    def __init__(self, net: torch.nn.Module, device: torch.device, gamma: float = 10):
        self.net = net
        self.device = device
        self.x_dim = 3
        self.u_dim = 3
        self.gamma = gamma

    @torch.no_grad()
    def sphere(
        self,
        u: torch.Tensor,
        env,
        return_b: bool = False,
    ) -> torch.Tensor:
        psm_pos = env._get_robot_state(0)[:3]
        center, radius = env.get_sphere_prop()
        
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        center = torch.from_numpy(center).float().unsqueeze(0).to(self.device)
        
        with torch.enable_grad():
            psm_pos.requires_grad_(True)
            b = torch.sum((psm_pos - center) ** 2) - radius ** 2
            b.backward()
            grad_b = psm_pos.grad.detach()

        if return_b:
            return b.item()
        # Reset requires_grad to False before using psm_pos further
        psm_pos.requires_grad_(False)
        
        # Obtain the dynamics
        net_out = self.net(psm_pos)  # [1, 12]
        fx = net_out[:, :self.x_dim]  # [1, 3]
        gx = net_out[:, self.x_dim:]  # [1, 9]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        # Compute Lie derivative
        Lfb = grad_b @ fx.T  # [1, 1]
        Lgb = grad_b @ gx.T  # [1, 9]
        
        gamma = self.gamma
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
    
    @torch.no_grad()
    def cylinder(
        self,
        u: torch.Tensor,
        env,
        return_b: bool = False,
    ) -> torch.Tensor:
        psm_pos = env._get_robot_state(0)[:3]
        cyl_center, cyl_axis, cyl_length, cyl_radius = env.get_cylinder_prop()
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().unsqueeze(0).to(self.device)
        cyl_axis = torch.from_numpy(-cyl_axis).float().unsqueeze(0).to(self.device)
        
        with torch.enable_grad():
            psm_pos.requires_grad_(True)

            # --- 1. Calculate Axial Barrier (b_axial) ---
            # Linear distance along the axis relative to the cylinder center.
            # b_axial > 0 if psm is above the top or below the bottom plane.
            axial = torch.dot((psm_pos - cyl_center).squeeze(), cyl_axis.squeeze())
            b_axial = torch.abs(axial) - cyl_length / 2.0

            # --- 2. Calculate Radial Barrier (b_radial) ---
            # Linear perpendicular distance from the axis minus the radius.
            # b_radial > 0 if psm is outside the radius.
            vec_from_axis_point = psm_pos - cyl_center
            radial_dist = torch.linalg.cross(vec_from_axis_point, cyl_axis).norm()
            b_radial = radial_dist - cyl_radius

            # --- 3. Combine Barriers with torch.max ---
            # The robot is safe if ANY of the following is true:
            # - above/below the cylinder caps, or
            # - outside the radius.
            # torch.max() implements this "OR" logic differentiably.
            # b will only be negative if inside radius AND between planes.
            b = torch.max(torch.stack([b_axial, b_radial]))
            #add return b(x)
            if return_b:
                return b.item()

            # --- 5. Compute Gradient ---
            # Backpropagate from the final combined barrier 'b'.
            # PyTorch automatically routes the gradient through the
            # correct function (b_axial or b_radial) that was the max.
            if b.grad_fn:
                # Clear old gradients before backward pass
                if psm_pos.grad is not None:
                    psm_pos.grad.zero_()
                    
                b.backward()
                grad_b = psm_pos.grad.detach()
            else:
                # Handle case where b is not part of a graph (e.g., inputs don't require grad)
                grad_b = torch.zeros_like(psm_pos)

        # Reset requires_grad to False before using psm_pos further
        psm_pos.requires_grad_(False)
        
        # Obtain the dynamics
        net_out = self.net(psm_pos)  # [1, 12]
        fx = net_out[:, :self.x_dim]  # [1, 3]
        gx = net_out[:, self.x_dim:]  # [1, 9]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        # Compute Lie derivative
        Lfb = grad_b @ fx.T  # [1, 1]
        Lgb = grad_b @ gx.T  # [1, 9]
        
        gamma = self.gamma
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
