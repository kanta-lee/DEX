import numpy as np
import torch
from cvxopt import solvers
from cvxopt.base import matrix

from surrol.tasks.gauze_retrieve_sphere import GauzeRetrieveSphere
from surrol.tasks.needle_pick_sphere import NeedlePickSphere
from surrol.tasks.gauze_retrieve_liver import GauzeRetrieveCylinder
from surrol.tasks.needle_pick_liver import NeedlePickCylinder


def qp_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    return np.array(sol['x']).flatten()


class CBF():
    def __init__(self, net: torch.nn.Module, device: torch.device):
        self.net = net
        self.device = device
        self.x_dim = 3
        self.u_dim = 3

    @torch.no_grad()
    def needle_pick_sphere(
        self,
        u: torch.Tensor,
        env: NeedlePickSphere,
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
        
        gamma = 1
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
    
    @torch.no_grad()
    def needle_pick_cylinder(
        self,
        u: torch.Tensor,
        env: NeedlePickCylinder,
    ) -> torch.Tensor:
        psm_pos = env._get_robot_state(0)[:3]
        cyl_center, cyl_axis, cyl_length, cyl_radius = env.get_cylinder_prop()
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().unsqueeze(0).to(self.device)
        cyl_axis = torch.from_numpy(-cyl_axis).float().unsqueeze(0).to(self.device)
        
        with torch.enable_grad():
            psm_pos.requires_grad_(True)
            
            # --- 1. Define Cylinder Top Plane ---
            # We define the top plane by a point on it (cyl_top_center)
            # and its normal vector (cyl_axis).
            cyl_top_center = cyl_center + cyl_length * cyl_axis

            # --- 2. Calculate Vertical Barrier (b_vertical) ---
            # This is the signed linear distance from psm_pos to the top plane.
            # We calculate this using the dot product, as defined in the paper.
            # b_vertical > 0 if psm is "above" the plane (in the direction of cyl_axis)
            # b_vertical < 0 if psm is "below" the plane (unsafe region)
            vec_to_top = psm_pos - cyl_top_center
            
            # Use .squeeze() to make the tensors 1D (shape [3]) for torch.dot
            b_vertical = torch.dot(vec_to_top.squeeze(), cyl_axis.squeeze())

            # --- 3. Calculate Radial Barrier (b_radial) ---
            # This is the squared perpendicular distance from psm_pos to the axis,
            # minus the squared radius.
            # b_radial > 0 if psm is outside the radius.
            # b_radial < 0 if psm is inside the radius (unsafe region).
            vec_from_axis_point = psm_pos - cyl_center
            radial_dist_sq = torch.linalg.cross(vec_from_axis_point, cyl_axis).norm().pow(2)
            b_radial = radial_dist_sq - cyl_radius ** 2

            # --- 4. Combine Barriers with torch.max ---
            # The robot is safe if EITHER b_vertical >= 0 OR b_radial >= 0.
            # torch.max() implements this "OR" logic differentiably.
            # b will only be negative if *both* are negative (inside radius AND below top).
            b = torch.max(b_vertical, b_radial)

            # --- 5. Compute Gradient ---
            # Backpropagate from the final combined barrier 'b'.
            # PyTorch automatically routes the gradient through the
            # correct function (b_vertical or b_radial) that was the max.
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
        
        gamma = 1
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
    
    @torch.no_grad()
    def gauze_retrieve_sphere(
        self,
        u: torch.Tensor,
        env: GauzeRetrieveSphere,
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
        
        gamma = 1
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
        
    @torch.no_grad()
    def gauze_retrieve_cylinder(
        self,
        u: torch.Tensor,
        env: GauzeRetrieveCylinder,
    ) -> torch.Tensor:
        psm_pos = env._get_robot_state(0)[:3]
        cyl_center, cyl_axis, cyl_length, cyl_radius = env.get_cylinder_prop()
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().unsqueeze(0).to(self.device)
        cyl_axis = torch.from_numpy(-cyl_axis).float().unsqueeze(0).to(self.device)
        
        with torch.enable_grad():
            psm_pos.requires_grad_(True)
            
            # --- 1. Define Cylinder Top Plane ---
            # We define the top plane by a point on it (cyl_top_center)
            # and its normal vector (cyl_axis).
            cyl_top_center = cyl_center + cyl_length * cyl_axis

            # --- 2. Calculate Vertical Barrier (b_vertical) ---
            # This is the signed linear distance from psm_pos to the top plane.
            # We calculate this using the dot product, as defined in the paper.
            # b_vertical > 0 if psm is "above" the plane (in the direction of cyl_axis)
            # b_vertical < 0 if psm is "below" the plane (unsafe region)
            vec_to_top = psm_pos - cyl_top_center
            
            # Use .squeeze() to make the tensors 1D (shape [3]) for torch.dot
            b_vertical = torch.dot(vec_to_top.squeeze(), cyl_axis.squeeze())

            # --- 3. Calculate Radial Barrier (b_radial) ---
            # This is the squared perpendicular distance from psm_pos to the axis,
            # minus the squared radius.
            # b_radial > 0 if psm is outside the radius.
            # b_radial < 0 if psm is inside the radius (unsafe region).
            vec_from_axis_point = psm_pos - cyl_center
            radial_dist_sq = torch.linalg.cross(vec_from_axis_point, cyl_axis).norm().pow(2)
            b_radial = radial_dist_sq - cyl_radius ** 2

            # --- 4. Combine Barriers with torch.max ---
            # The robot is safe if EITHER b_vertical >= 0 OR b_radial >= 0.
            # torch.max() implements this "OR" logic differentiably.
            # b will only be negative if *both* are negative (inside radius AND below top).
            b = torch.max(b_vertical, b_radial)

            # --- 5. Compute Gradient ---
            # Backpropagate from the final combined barrier 'b'.
            # PyTorch automatically routes the gradient through the
            # correct function (b_vertical or b_radial) that was the max.
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
        
        gamma = 1
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = qp_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u