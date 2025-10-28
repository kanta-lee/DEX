import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint
from math import sqrt, cos, sin
from cvxopt import solvers, matrix
from scipy.spatial.transform import Rotation

# from ..SurRoL.surrol.tasks.reach_hemisphere import Reach
from surrol.tasks.reach_hemisphere import Reach
from surrol.tasks.hemipuncture import Hemipuncture
# from ..SurRoL.surrol.tasks.hemipuncture import Hemipuncture
from surrol.utils.pybullet_utils import get_link_pose

# # Load dataset
# obs = np.load('./data/obs3.npy')  # [100, 51, 19]
# obs = torch.tensor(obs).float()
# SCALING = 5.0
# acs = np.load('./data/acs3.npy')  # [100, 50 ,5]
# acs = acs * 0.01 * SCALING  # [100, 50, 3]
# acs = torch.tensor(acs).float()

# # Training data
# x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
# u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 3]

# # Testing data
# x_test = x_train[-1, :, :, :]  # [51, 1, 3]
# u_test = u_train[-1, :, :, :]  # [50, 1, 3]

# # Initial condition for testing
# x_test0 = x_train[-1, 0, :, :]  # [1, 3]
# u_test0 = u_train[-1, 0, :, :]  # [1, 3]


def cvx_solver(P, q, G, h):
    mat_P = matrix(P.cpu().numpy())
    mat_q = matrix(q.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)

    # Convert solution to numpy array
    return np.array(sol['x']).flatten()


class CBF(nn.Module):

    def __init__(self, fc_param):
        super(CBF, self).__init__()

        self.net = self.build_mlp(fc_param)
        self.x_dim = fc_param[0]
        self.u_dim = (fc_param[-1] - fc_param[0]) // fc_param[0]

        # Initializing weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.u = None
        self.device = torch.device(
            'cuda:' + str(0)
            if torch.cuda.is_available() else 'cpu'
        )

        # constraint
        # val = 5 * URDF's val

    def forward(self, t, x):
        if self.training:
            # x.shape = [20, 1, 3]
            net_out = self.net(x)  # [20, 1, 12]
            fx = net_out[:, :, :self.x_dim]
            gx = net_out[:, :, self.x_dim:]
            
            # Reshape gx for batch matrix multiplication
            gx = gx.view(x.shape[0], self.u_dim, self.x_dim)  # [20, 3, 3]

            # \dot{x} = f(x) + g(x) * u
            out = fx + torch.matmul(self.u, gx) # [20, 1, 3]
        else:
            # For test and evaluation, x.shape = [1, 3]
            net_out = self.net(x)  # [1, 12]
            fx = net_out[:, :self.x_dim]  # [1, 3]
            gx = net_out[:, self.x_dim:]  # [1, 9]
            gx = torch.reshape(gx, (self.u_dim, self.x_dim))
            out = fx + self.u @ gx

        return out

    def build_discretized_center_line(self, cylinder_length, radius, center, cylinder_ori):
        rot_matrix = Rotation.from_quat(np.array(cylinder_ori)).as_matrix()

        # discretize the center line
        all_point = []
        num = 100
        for i in range(num):
            ori_xyz = np.array([0, -radius, radius + i / num * cylinder_length]).reshape(3, 1)
            all_point.append((rot_matrix@ori_xyz).reshape(-1)+np.array(center))
        for i in range(num):
            theta = i / num * (np.pi/2)
            ori_xyz = np.array([0, -radius*sin(theta), radius*(1-cos(theta))]).reshape(3, 1)
            all_point.append((rot_matrix @ ori_xyz).reshape(-1)+np.array(center))
        for i in range(num):
            theta = i / num * (np.pi/2)
            ori_xyz = np.array([0, radius*sin(theta), radius*(cos(theta)-1)]).reshape(3, 1)
            all_point.append((rot_matrix @ ori_xyz).reshape(-1)+np.array(center))
        for i in range(num):
            ori_xyz = np.array([0, radius, -radius - i / num * cylinder_length]).reshape(3, 1)
            all_point.append((rot_matrix @ ori_xyz).reshape(-1)+np.array(center))
        # 400, 3
        self.all_point = np.stack(all_point, axis=0)

    def constraint_valid(self, constraint_type, robot,
                         constraint_center=None, length=None,
                         point=None, normal_vector=None, radius=None, ori_vector=None):
        # Assign robot state
        x, y, z = robot[0], robot[1], robot[2]

        if constraint_type == 1:
            # NOTE: Originally for sphere
            # x0, y0, z0 = constraint_center
            # b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - radius ** 2
            # violate = (b <= 0)
            proj_vec = np.dot(np.array(ori_vector), np.array(robot)-np.array(constraint_center))*np.array(ori_vector)
            norm_vec = np.array(robot)-(np.array(constraint_center)+proj_vec)
            violate = (np.sum(norm_vec**2)-radius**2 > 0)
        elif constraint_type == 2:
            x0, y0, z0 = point
            a0, b0, c0 = normal_vector
            norm_score = sqrt(a0 ** 2 + b0 ** 2 + c0 ** 2)
            a0, b0, c0 = a0 / norm_score, b0 / norm_score, c0 / norm_score
            b = (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) ** 2 - length ** 2
            violate = (b <= 0)
        elif constraint_type == 3:
            x0, y0, z0 = constraint_center
            b = (x - x0) ** 2 + (y - y0) ** 2 - radius ** 2
            violate = (b <= 0)
        elif constraint_type == 4:
            x0, y0, z0 = constraint_center
            b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - radius ** 2
            violate = (b > 0)
        elif constraint_type == 5:
            proj_vec = np.dot(np.array(ori_vector), np.array(robot)-np.array(constraint_center))*np.array(ori_vector)
            norm_vec = np.array(robot)-(np.array(constraint_center)+proj_vec)
            violate = (np.sum(norm_vec**2)-radius**2 > 0)
        elif constraint_type == 6:
            dis = np.sum((self.all_point-np.array(robot))**2, axis=-1)
            min_dis_ind = np.argmin(dis)
            norm_vec = np.array(robot)-self.all_point[min_dis_ind]
            violate = (np.sum(norm_vec**2)-radius**2 > 0)
        return violate

    def dCBF_sphere(self, robot, u, f, g1, g2, g3, constraint_center, radius):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        # x0, y0, z0 = 2.66255212, -0.00543937, 3.49126458
        x0, y0, z0 = constraint_center

        r = radius

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2

        Lfb = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        Lgb = 2 * (x - x0) * g1 \
            + 2 * (y - y0) * g2 \
            + 2 * (z - z0) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def dCBF_surface(self, robot, u, f, g1, g2, g3, point, normal_vector, length):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle is a surface defined by a point on the surface and the normal vector
        x0, y0, z0 = point
        a0, b0, c0 = normal_vector
        norm_score = sqrt(a0**2+b0**2+c0**2)
        a0, b0, c0 = a0/norm_score, b0/norm_score, c0/norm_score

        d = length

        # Compute barrier function
        b = (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) ** 2 - d ** 2

        Lfb = 2 * a0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 0] \
            + 2 * b0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 1] \
            + 2 * c0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * f[0, 2]

        Lgb = 2 * a0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g1 \
            + 2 * b0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g2 \
            + 2 * c0 * (a0 * (x - x0) + b0 * (y - y0) + c0 * (z - z0)) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def dCBF_plate(self, robot, u, f, g1, g2, g3, plate_center, radius, length, current_area):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # box center
        x0, y0, z0 = plate_center

        r = radius
        d = length

        # Compute barrier function
        if current_area == 1:
            b = (x - x0) ** 2 + (y - y0) ** 2 - r ** 2

            Lfb = 2 * (x - x0) * f[0, 0] \
                + 2 * (y - y0) * f[0, 1]

            Lgb = 2 * (x - x0) * g1 \
                + 2 * (y - y0) * g2

        elif current_area == 2:
            b = (z - z0) ** 2 - (d/2) ** 2

            Lfb = 2 * (z - z0) * f[0, 2]
            Lgb = 2 * (z - z0) * g3

        elif current_area == 3:
            b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - (r ** 2 + (d / 2) ** 2)

            Lfb = 2 * (x - x0) * f[0, 0] \
                  + 2 * (y - y0) * f[0, 1] \
                  + 2 * (z - z0) * f[0, 2]

            Lgb = 2 * (x - x0) * g1 \
                  + 2 * (y - y0) * g2 \
                  + 2 * (z - z0) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe  # [1, 3]
        h = b_safe.unsqueeze(0)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def dCBF_half_sphere(self, robot, u, f, g1, g2, g3, center, radius, current_area):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        x0, y0, z0 = center

        r = radius

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2

        Lfb = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        Lgb = 2 * (x - x0) * g1 \
            + 2 * (y - y0) * g2 \
            + 2 * (z - z0) * g3

        gamma = 1
        if current_area == 1:
            b_safe = Lfb + gamma * b
            A_safe = -Lgb
        elif current_area == 2:
            b_safe = -(Lfb + gamma * b)
            A_safe = Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def dCBF_cylinder(self, robot, u, f, g1, g2, g3, ori_vec, center, radius, current_area):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        x0, y0, z0 = center
        #  cylinder orientation vector
        orix, oriy, oriz = ori_vec

        r = radius

        # proj_factor = orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)
        # norm_vec_x = x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix
        # norm_vec_y = y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy
        # norm_vec_z = z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz

        # Compute barrier function
        # derivation
        # b = (x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix) ** 2 +\
        #     (y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy) ** 2 +\
        #     (z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * (oriy * (y - y0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriy ** 2)(y - y0) - oriy * (orix * (x - x0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriz ** 2)(z - z0) - oriz * (orix * (x - x0) + oriy * (y - y0))) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * oriy * (y - y0) - orix * oriz * (z - z0)) ** 2 +\
        #     (- oriy * orix * (x - x0) + (1 - oriy ** 2)(y - y0) - oriy * oriz * (z - z0)) ** 2 +\
        #     (- oriz * orix * (x - x0) - oriz * oriy * (y - y0) + (1 - oriz ** 2)(z - z0)) ** 2 - r ** 2
        c1x = (1 - orix ** 2)
        c1y = - orix * oriy
        c1z = - orix * oriz
        c2x = - oriy * orix
        c2y = (1 - oriy ** 2)
        c2z = - oriy * oriz
        c3x = - oriz * orix
        c3y = - oriz * oriy
        c3z = (1 - oriz ** 2)
        b = (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) ** 2 +\
            (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) ** 2 +\
            (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0)) ** 2 - r ** 2

        Lfb = (2 * c1x * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2x * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3x * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 0] \
            + (2 * c1y * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2y * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3y * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 1] \
            + (2 * c1z * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2z * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3z * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 2]

        Lgb = (2 * c1x * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2x * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3x * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g1 \
            + (2 * c1y * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2y * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3y * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g2 \
            + (2 * c1z * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2z * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3z * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g3

        gamma = 1
        if current_area == 1:
            b_safe = Lfb + gamma * b
            A_safe = -Lgb
        elif current_area == 2:
            b_safe = -(Lfb + gamma * b)
            A_safe = Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def dCBF_cylinder(self, robot, u, f, g1, g2, g3, ori_vec, center, radius):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        x0, y0, z0 = center
        # Cylinder orientation vector
        orix, oriy, oriz = ori_vec

        r = radius

        # proj_factor = orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)
        # norm_vec_x = x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix
        # norm_vec_y = y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy
        # norm_vec_z = z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz

        # Compute barrier function
        # derivation
        # b = (x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix) ** 2 +\
        #     (y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy) ** 2 +\
        #     (z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * (oriy * (y - y0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriy ** 2)(y - y0) - oriy * (orix * (x - x0) + oriz * (z - z0))) ** 2 +\
        #     ((1 - oriz ** 2)(z - z0) - oriz * (orix * (x - x0) + oriy * (y - y0))) ** 2 - r ** 2
        # b = ((1 - orix ** 2)(x - x0) - orix * oriy * (y - y0) - orix * oriz * (z - z0)) ** 2 +\
        #     (- oriy * orix * (x - x0) + (1 - oriy ** 2)(y - y0) - oriy * oriz * (z - z0)) ** 2 +\
        #     (- oriz * orix * (x - x0) - oriz * oriy * (y - y0) + (1 - oriz ** 2)(z - z0)) ** 2 - r ** 2
        c1x = (1 - orix ** 2)
        c1y = - orix * oriy
        c1z = - orix * oriz
        c2x = - oriy * orix
        c2y = (1 - oriy ** 2)
        c2z = - oriy * oriz
        c3x = - oriz * orix
        c3y = - oriz * oriy
        c3z = (1 - oriz ** 2)
        b = (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) ** 2 +\
            (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) ** 2 +\
            (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0)) ** 2 - r ** 2

        Lfb = (2 * c1x * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2x * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3x * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 0] \
            + (2 * c1y * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2y * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3y * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 1] \
            + (2 * c1z * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2z * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3z * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * f[0, 2]

        Lgb = (2 * c1x * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2x * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3x * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g1 \
            + (2 * c1y * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2y * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3y * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g2 \
            + (2 * c1z * (c1x * (x - x0) + c1y * (y - y0) + c1z * (z - z0)) +
               2 * c2z * (c2x * (x - x0) + c2y * (y - y0) + c2z * (z - z0)) +
               2 * c3z * (c3x * (x - x0) + c3y * (y - y0) + c3z * (z - z0))) * g3

        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb
        # if current_area == 1:
        #     b_safe = Lfb + gamma * b
        #     A_safe = -Lgb
        # elif current_area == 2:
        #     b_safe = -(Lfb + gamma * b)
        #     A_safe = Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out 

    def dCBF_complex_cylinder(self, robot, u, f, g1, g2, g3, radius, current_area):
        """Enforce CBF on action

        Args:
            robot  ([1, 3]): robot state
            u  ([1, 3]): action
            f  ([1, 3]): fx
            g1 ([1, 3]): first row of gx
            g2 ([1, 3]): second row of gx
            g3 ([1, 3]): third row of gx
        """
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Find the center line point
        dis = np.sum((self.all_point - np.array([x.item(), y.item(), z.item()])) ** 2, axis=-1)
        min_dis_ind = np.argmin(dis)
        x0, y0, z0 = self.all_point[min_dis_ind].tolist()

        r = radius

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2

        Lfb = 2 * (x - x0) * f[0, 0] \
            + 2 * (y - y0) * f[0, 1] \
            + 2 * (z - z0) * f[0, 2]

        Lgb = 2 * (x - x0) * g1 \
            + 2 * (y - y0) * g2 \
            + 2 * (z - z0) * g3

        gamma = 1
        if current_area == 1:
            b_safe = Lfb + gamma * b
            A_safe = -Lgb
        elif current_area == 2:
            b_safe = -(Lfb + gamma * b)
            A_safe = Lgb

        dim = g1.shape[1]  # = 3
        G = A_safe.to(self.device)
        h = b_safe.unsqueeze(0).to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def build_mlp(self, filters, no_act_last_layer=True, activation='gelu'):
        if activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError(
                f'Not supported activation function {activation}')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters)-2):
                modules.append(activation)

        modules = nn.Sequential(*modules)
        return modules

    def dCBF_cylinder_2(self, robot, remote_center, u, f, g, ori_vec, center, radius):
        # assert robot.shape == (1, self.x_dim)
        # assert u.shape == (1, self.u_dim)
        # assert f.shape == (1, self.x_dim)
        # assert g.shape == (1, self.x_dim * self.u_dim)
        # assert len(center) == 3 # List
        # assert len(ori_vec) == 3 # List
        # assert radius > 0
        # remote_center: np.array, size 3
        
        # "ori_vec" is a unit vector.
        
        # Obstacle point position
        x0, y0, z0 = center
        
        # PSM1
        x1, y1, z1 = robot[0, 0], robot[0, 1], robot[0, 2]
        
        # Remote center
        xR, yR, zR = remote_center[0], remote_center[1], remote_center[2]
        
        # Point on the stick
        x2 = x1 + 0.07 * (xR - x1)
        y2 = y1 + 0.07 * (yR - y1)
        z2 = z1 + 0.07 * (zR - z1)
        
        # Cylinder orientation vector
        orix, oriy, oriz = ori_vec

        r = radius

        '''
        proj_factor = orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)
        norm_vec_x = x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix
        norm_vec_x = x - (x0 + proj_factor * orix)
        norm_vec_y = y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy
        norm_vec_z = z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz

        Compute barrier function
        derivation
        b = (x - x0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * orix) ** 2 +\
            (y - y0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriy) ** 2 +\
            (z - z0 - (orix * (x - x0) + oriy * (y - y0) + oriz * (z - z0)) * oriz) ** 2 - r ** 2
        b = ((1 - orix ** 2)(x - x0) - orix * (oriy * (y - y0) + oriz * (z - z0))) ** 2 +\
            ((1 - oriy ** 2)(y - y0) - oriy * (orix * (x - x0) + oriz * (z - z0))) ** 2 +\
            ((1 - oriz ** 2)(z - z0) - oriz * (orix * (x - x0) + oriy * (y - y0))) ** 2 - r ** 2
        b = ((1 - orix ** 2)(x - x0) - orix * oriy * (y - y0) - orix * oriz * (z - z0)) ** 2 +\
            (- oriy * orix * (x - x0) + (1 - oriy ** 2)(y - y0) - oriy * oriz * (z - z0)) ** 2 +\
            (- oriz * orix * (x - x0) - oriz * oriy * (y - y0) + (1 - oriz ** 2)(z - z0)) ** 2 - r ** 2
        '''
        
        # Coefficient for norm_vec_x
        c1x = (1 - orix ** 2)
        c1y = - orix * oriy
        c1z = - orix * oriz
        # Coefficient for norm_vec_y
        c2x = - oriy * orix
        c2y = (1 - oriy ** 2)
        c2z = - oriy * oriz
        # Coefficient for norm_vec_z
        c3x = - oriz * orix
        c3y = - oriz * oriy
        c3z = (1 - oriz ** 2)
        
        b1 = (
            (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) ** 2 +
            (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) ** 2 +
            (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
        
        db1 = torch.tensor([
            2 * c1x * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2x * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3x * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1y * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2y * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3y * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0)) ,
            
            2 * c1z * (c1x * (x1 - x0) + c1y * (y1 - y0) + c1z * (z1 - z0)) +
            2 * c2z * (c2x * (x1 - x0) + c2y * (y1 - y0) + c2z * (z1 - z0)) +
            2 * c3z * (c3x * (x1 - x0) + c3y * (y1 - y0) + c3z * (z1 - z0))
        ]).unsqueeze(0).to(self.device)
        
        b2 = (
            (c1x * (x2 - x0) + c1y * (y2 - y0) + c1z * (z2 - z0)) ** 2 + \
            (c2x * (x2 - x0) + c2y * (y2 - y0) + c2z * (z2 - z0)) ** 2 + \
            (c3x * (x2 - x0) + c3y * (y2 - y0) + c3z * (z2 - z0)) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
        
        db2 = torch.tensor([
            2 * c1x * (c1x * (x2 - x0) + c1y * (y2 - y0) + c1z * (z2 - z0)) * 0.93 +
            2 * c2x * (c2x * (x2 - x0) + c2y * (y2 - y0) + c2z * (z2 - z0)) * 0.93 +
            2 * c3x * (c3x * (x2 - x0) + c3y * (y2 - y0) + c3z * (z2 - z0)) * 0.93 ,
            
            2 * c1y * (c1x * (x2 - x0) + c1y * (y2 - y0) + c1z * (z2 - z0)) * 0.93 +
            2 * c2y * (c2x * (x2 - x0) + c2y * (y2 - y0) + c2z * (z2 - z0)) * 0.93 +
            2 * c3y * (c3x * (x2 - x0) + c3y * (y2 - y0) + c3z * (z2 - z0)) * 0.93 ,
        
            2 * c1z * (c1x * (x2 - x0) + c1y * (y2 - y0) + c1z * (z2 - z0)) * 0.93 +
            2 * c2z * (c2x * (x2 - x0) + c2y * (y2 - y0) + c2z * (z2 - z0)) * 0.93 +
            2 * c3z * (c3x * (x2 - x0) + c3y * (y2 - y0) + c3z * (z2 - z0)) * 0.93
        ]).unsqueeze(0).to(self.device)
        
        Lfb1 = db1 @ f.T
        Lfb2 = db2 @ f.T
        
        g = torch.reshape(g, (self.u_dim, self.x_dim))
        Lgb1 = db1 @ g.T
        Lgb2 = db2 @ g.T

        Lfb = torch.cat([Lfb1, Lfb2], dim=0)
        Lgb = torch.cat([Lgb1, Lgb2], dim=0)
        b = torch.cat([b1, b2], dim=0)
        
        gamma = 1
        A_safe = -Lgb
        b_safe = Lfb + gamma * b

        dim = self.u_dim
        G = A_safe.to(self.device)
        h = b_safe.to(self.device)
        P = torch.eye(dim).to(self.device)
        q = -u.T

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out
    
    def dCBF_sphere_2(self, robot, remote_center, u, f, g, center, radius):

        # Obstacle point position
        x0, y0, z0 = center
        
        # PSM1
        x1, y1, z1 = robot[0, 0], robot[0, 1], robot[0, 2]
        
        # Remote center
        xR, yR, zR = remote_center[0], remote_center[1], remote_center[2]
        
        # Point on the stick
        x2 = x1 + 0.07 * (xR - x1)
        y2 = y1 + 0.07 * (yR - y1)
        z2 = z1 + 0.07 * (zR - z1)
        
        r = radius

        # Compute barrier function
        b1 = (
            (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
        
        db1 = torch.tensor([
            2 * (x1 - x0),
            2 * (y1 - y0),
            2 * (z1 - z0)
        ]).unsqueeze(0).to(self.device)
        
        b2 = (
            (x2 - x0) ** 2 + (y2 - y0) ** 2 + (z2 - z0) ** 2 - r ** 2
        ).clone().detach().reshape(1, 1).to(self.device)
        
        db2 = torch.tensor([
            2 * (x2 - x0) * 0.93,
            2 * (y2 - y0) * 0.93,
            2 * (z2 - z0) * 0.93
        ]).unsqueeze(0).to(self.device)
        
        Lfb1 = db1 @ f.T
        Lfb2 = db2 @ f.T

        g = torch.reshape(g, (self.u_dim, self.x_dim))
        Lgb1 = db1 @ g.T
        Lgb2 = db2 @ g.T
        
        Lfb = torch.cat([Lfb1, Lfb2], dim=0)
        Lgb = torch.cat([Lgb1, Lgb2], dim=0)
        b = torch.cat([b1, b2], dim=0)
        
        gamma = 1
        b_safe = Lfb + gamma * b
        A_safe = -Lgb

        dim = self.u_dim
        G = A_safe.to(self.device)
        h = b_safe.to(self.device)
        P = torch.eye(dim).to(self.device)
        q = -u.T

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())
        
        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out
    
    def dCBF_hemisphere_2(self, robot, u, f, g, center, radius):
        
        # Assign robot state
        x, y, z = robot[0, 0], robot[0, 1], robot[0, 2]

        # Obstacle point position
        x0, y0, z0 = center

        r = radius

        # Compute barrier function
        b = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2
        
        db = torch.tensor([
            2 * (x - x0),
            2 * (y - y0),
            2 * (z - z0)
        ]).unsqueeze(0).to(self.device)
        
        Lfb = db @ f.T
        
        g = torch.reshape(g, (self.u_dim, self.x_dim))
        Lgb = db @ g.T

        gamma = 1
        # The sign is difference. We want b <= 0
        b_safe = -(Lfb + gamma * b)
        A_safe = Lgb

        dim = self.u_dim  # = 3
        G = A_safe.to(self.device)
        h = b_safe.to(self.device)  # [1, 1]
        P = torch.eye(dim).to(self.device)  # [3, 3]
        q = -u.T  # [3, 1]

        # NOTE: different x from above now
        x = cvx_solver(P.double(), q.double(), G.double(), h.double())

        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(self.device)
        out = out.unsqueeze(0)
        return out

    def hollow_cylinder(self, robot, u, center, cyl_axis, radius):
        # Extract PSM position
        x = robot[0, :]
        
        # Compute distance square from PSM to cylinder axis
        dist = torch.linalg.cross(x - center, cyl_axis).norm()
        dist_sq = dist ** 2
        
        if abs(dist - radius) < 1e-4:
            raise ValueError("Too close to the wall!")
        
        is_inside = dist < radius
        
        # Barrier function
        b = radius ** 2 - dist_sq if is_inside else dist_sq - radius ** 2
        
        grad_b = -2 * torch.linalg.cross(cyl_axis, torch.linalg.cross(x - center, cyl_axis))
        grad_b = grad_b if is_inside else -grad_b
        grad_b = grad_b.unsqueeze(0)
        
        # Obtain the dynamics
        net_out = self.net(robot)  # [1, 12]
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

        modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u

    def hemisphere(self, robot, u, center, radius):
        # Extract PSM position
        x = robot[0, :]
        
        # Barrier function
        # FIXME: Python cannot call norm() on numpy array
        # b = np.linalg.norm(x - center) ** 2 - radius ** 2
        b = (x - center).norm() ** 2 - radius ** 2
        
        if b < 0:
            raise ValueError("PSM already across the wall (unsafe)!")
        
        grad_b = 2 * (x - center).unsqueeze(0)
        
        # Obtain the dynamics
        net_out = self.net(robot)  # [1, 12]
        fx = net_out[:, :self.x_dim]  # [1, 3]
        gx = net_out[:, self.x_dim:]  # [1, 9]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        # Compute Lie derivative
        Lfb = grad_b @ fx.T  # [1, 1]
        Lgb = grad_b @ gx.T  # [1, 9]
        
        gamma = 1
        G = Lgb.to(self.device)
        h = -(Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T

        modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u

    def hollow_hemisphere(self, robot, u, center, radius):
        # Extract PSM position
        x = robot[0, :]
        
        # Compute distance square from PSM to sphere center
        dist_sq = (x - center).norm() ** 2
        
        if abs(dist_sq - radius ** 2) < 1e-4:
            raise ValueError("Too close to the wall!")
        
        is_inside = dist_sq < radius ** 2
        
        # Barrier function
        b = radius ** 2 - dist_sq if is_inside else dist_sq - radius ** 2
        
        grad_b = -2 * (x - center).unsqueeze(0) if is_inside else 2 * (x - center).unsqueeze(0)
        
        # Obtain the dynamics
        net_out = self.net(robot)  # [1, 12]
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

        modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
    
    @torch.no_grad()
    def reach(
        self,
        psm_pos: np.ndarray,
        u: torch.Tensor,
        env: Reach,
    ) -> torch.Tensor:
        
        assert psm_pos.shape == (self.x_dim,), "The 'psm_pos' array must have shape (3,)."
        assert u.shape == (1, self.u_dim), "The 'u' tensor must have shape (1, 3)."

        current_region = env.get_current_region(psm_pos)
        cyl_center, cyl_axis, _, cyl_radius = env.get_cylinder_prop()
        hemi_center, hemi_radius = env.get_hemisphere_prop()
        
        print("Current:", current_region)
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().to(self.device)
        hemi_center = torch.from_numpy(hemi_center).float().to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().to(self.device)
        cyl_axis = torch.from_numpy(cyl_axis).float().to(self.device)

        if current_region == env.Region.OUTSIDE:
            # Compute distance square from PSM to hemisphere center
            dist_hemi_sq = torch.norm(psm_pos - hemi_center) ** 2
            
            if torch.abs(dist_hemi_sq - hemi_radius ** 2) < 1e-4:
                raise ValueError("Too close to the wall!")

            # Barrier function
            b_hemi = (dist_hemi_sq - hemi_radius ** 2).view(1, 1)
            grad_b_hemi = 2 * torch.unsqueeze(psm_pos - hemi_center, 0)
            
            assert b_hemi.shape == (1, 1)
            assert grad_b_hemi.shape == (1, 3)
            
            b = b_hemi
            grad_b = grad_b_hemi

            if env.is_in_cylinder_region(psm_pos.cpu().numpy()):
                # Compute distance square from PSM to cylinder axis
                dist_cyl_sq = torch.linalg.cross(psm_pos - cyl_center, cyl_axis).norm() ** 2

                if torch.abs(dist_cyl_sq - cyl_radius ** 2) < 1e-4:
                    raise ValueError("Too close to the wall!")

                # Barrier function
                b_cyl = (dist_cyl_sq - cyl_radius ** 2).view(1, 1)
                grad_b_cyl = 2 * torch.linalg.cross(torch.linalg.cross(psm_pos - cyl_center, cyl_axis), cyl_axis).unsqueeze(0)
                
                b = torch.vstack([b, b_cyl])
                grad_b = torch.vstack([grad_b, grad_b_cyl])

                assert b.shape == (2, 1), f"Unexpected barrier shape: {b.shape}"
                assert grad_b.shape == (2, 3), f"Unexpected barrier gradient shape: {grad_b.shape}"
                
        elif current_region == env.Region.INSIDE_CYLINDER_ONLY:
            # Compute distance square from PSM to cylinder axis
            dist_cyl_sq = torch.linalg.cross(psm_pos - cyl_center, cyl_axis).norm() ** 2

            if torch.abs(dist_cyl_sq - cyl_radius ** 2) < 1e-4:
                raise ValueError("Too close to the wall!")

            # Barrier function
            b = (cyl_radius ** 2 - dist_cyl_sq).view(1, 1)
            grad_b = -2 * torch.linalg.cross(torch.linalg.cross(psm_pos - cyl_center, cyl_axis), cyl_axis).unsqueeze(0)

            assert b.shape == (1, 1), f"Unexpected barrier shape: {b.shape}"
            assert grad_b.shape == (1, 3), f"Unexpected barrier gradient shape: {grad_b.shape}"

            # NOTE: No need to consider hemisphere here, since we want it to go inside.
        
        elif current_region == env.Region.INSIDE_HEMISPHERE:
            # Compute distance square from PSM to hemisphere center
            dist_hemi_sq = torch.norm(psm_pos - hemi_center) ** 2

            if torch.abs(dist_hemi_sq - hemi_radius ** 2) < 1e-4:
                raise ValueError("Too close to the wall!")

            # FIXME: What if we need to consider another point on the stick?
            #        Currently we only consider the tip point.

            # Barrier function
            b = (hemi_radius ** 2 - dist_hemi_sq).view(1, 1)
            grad_b = -2 * torch.unsqueeze(psm_pos - hemi_center, 0)

            assert b.shape == (1, 1), f"Unexpected barrier shape: {b.shape}"
            assert grad_b.shape == (1, 3), f"Unexpected barrier gradient shape: {grad_b.shape}"

        
        # Obtain the dynamics
        net_out = self.net(psm_pos.unsqueeze(0))
        assert net_out.shape == (1, self.x_dim + self.x_dim * self.u_dim), f"Unexpected network output shape: {net_out.shape}"
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        # Compute Lie derivative
        Lfb = grad_b @ fx.T
        Lgb = grad_b @ gx.T
        
        assert Lfb.shape == (b.shape[0], 1)
        assert Lgb.shape == (b.shape[0], self.u_dim)
        
        gamma = 1
        G = -Lgb.to(self.device)
        h = (Lfb + gamma * b).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T
        
        modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u

    @torch.no_grad()
    def hemipuncture_clf(
        self,
        u: torch.Tensor,
        env: Hemipuncture,
        method: int = 2
    ) -> torch.Tensor:
        assert method in range(1, 4 + 1), f"Invalid method selected. Choose either {list(range(1, 4 + 1))}."
        assert u.shape == (1, self.u_dim), "The 'u' tensor must have shape (1, 3)."
        
        psm_pos = env._get_robot_state(0)[:3]
        pitch_pos = np.array(get_link_pose(env.psm1.body, 4)[0])
        cyl_center, cyl_axis, _, _ = env.get_cylinder_prop()
        rc_pos = np.array(get_link_pose(env.psm1.body, 13)[0])
        stick_pos = pitch_pos + 0.1 * (rc_pos - pitch_pos)
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        pitch_pos = torch.from_numpy(pitch_pos).float().unsqueeze(0).to(self.device)
        stick_pos = torch.from_numpy(stick_pos).float().unsqueeze(0).to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().unsqueeze(0).to(self.device)
        cyl_axis = torch.from_numpy(-cyl_axis).float().unsqueeze(0).to(self.device)
        assert psm_pos.shape == (1, 3), f"Unexpected psm_pos shape: {psm_pos.shape}"
        assert pitch_pos.shape == (1, 3), f"Unexpected pitch_pos shape: {pitch_pos.shape}"
        assert stick_pos.shape == (1, 3), f"Unexpected stick_pos shape: {stick_pos.shape}"
        assert cyl_center.shape == (1, 3), f"Unexpected cyl_center shape: {cyl_center.shape}"
        assert cyl_axis.shape == (1, 3), f"Unexpected cyl_axis shape: {cyl_axis.shape}"
        
        # Obtain the dynamics
        if method >= 2:
            state = torch.cat([psm_pos, pitch_pos], dim=1)
            assert state.shape == (1, 6), f"Unexpected state shape: {state.shape}"
        else:
            state = psm_pos
            
        net_out = self.net(state)
        assert net_out.shape == (1, self.x_dim + self.x_dim * self.u_dim), f"Unexpected network output shape: {net_out.shape}"
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        # NOTE: Previous Trials
        # 1. Fixed target position at the center of cylinder is a bad idea because original
        #    RL action is already moving toward center direction therefore CLF-QP is ineffective.
        # 2. Define V as distance from the end-effector perpendicularly towards the nearest point 
        #    on the axis. However, projecting the "predicted next" position doesn't help as much. 
        #    Don't know mathematically why.
        # 3. Define V as perpendicular distance at current time step works well.
        
        V = torch.tensor([]).to(self.device)
        grad_V = torch.tensor([]).to(self.device)
        num_points = 0
        
        # =============================================================================
        #                                   Method 1
        # =============================================================================
        # Preserve this code for running old state of shape (1, 3)
        if state.shape == (1, 3):
            V_psm = 0.5 * torch.norm(torch.linalg.cross(psm_pos - cyl_center, cyl_axis)) ** 2
            print("V:", V_psm.item())
            V_psm = V_psm.view(1, 1)
            grad_V_psm = torch.linalg.cross(cyl_axis, torch.linalg.cross(psm_pos - cyl_center, cyl_axis))
            V = torch.cat([V, V_psm], dim=0)
            grad_V = torch.cat([grad_V, grad_V_psm], dim=0)
            num_points += 1
            assert V.shape == (num_points, 1), f"Unexpected V shape: {V.shape}"
            assert grad_V.shape == (num_points, 3), f"Unexpected grad_V shape: {grad_V.shape}"
            
        
        # =============================================================================
        #                                   Method 2
        # =============================================================================
        # Same as Method 1 but take into account the newly trained network with state of
        # 6 dimensions, now it doesn't skip cylinder region!
        if method >= 2 and state.shape == (1, 6) and env.get_region(psm_pos) != env.INSIDE_HEMISPHERE:
            V_psm = 0.5 * torch.norm(torch.linalg.cross(psm_pos - cyl_center, cyl_axis)) ** 2
            V_psm = V_psm.view(1, 1)
            grad_V_psm = torch.linalg.cross(cyl_axis, torch.linalg.cross(psm_pos - cyl_center, cyl_axis))
            grad_V_psm = torch.cat([grad_V_psm, torch.zeros(1, 3).to(self.device)], dim=1)
            V = torch.cat([V, V_psm], dim=0)
            grad_V = torch.cat([grad_V, grad_V_psm], dim=0)
            num_points += 1
            assert V.shape == (num_points, 1), f"Unexpected V shape: {V.shape}"
            assert grad_V.shape == (num_points, 6), f"Unexpected grad_V shape: {grad_V.shape}"
            # print("V:", V)
            # print("grad_V:", grad_V)
        
        # =============================================================================
        #                                   Method 3
        # =============================================================================
        # Try to make "tool_pitch" to move along axis as well.
        if method >= 3 and state.shape == (1, 6) and env.get_region(pitch_pos) != env.INSIDE_HEMISPHERE:
            V_pitch = 0.5 * torch.norm(torch.linalg.cross(pitch_pos - cyl_center, cyl_axis)) ** 2
            V_pitch = V_pitch.view(1, 1)
            grad_V_pitch = torch.linalg.cross(cyl_axis, torch.linalg.cross(pitch_pos - cyl_center, cyl_axis))
            grad_V_pitch = torch.cat([torch.zeros(1, 3).to(self.device), grad_V_pitch], dim=1)
            V = torch.cat([V, V_pitch], dim=0)
            grad_V = torch.cat([grad_V, grad_V_pitch], dim=0)
            num_points += 1
            assert V.shape == (num_points, 1), f"Unexpected V shape: {V.shape}"
            assert grad_V.shape == (num_points, 6), f"Unexpected grad_V shape: {grad_V.shape}"
            # print("Vcdad_V)
            
        # =============================================================================
        #                                   Method 4
        # =============================================================================
        # Try to make "stick" to move along axis as well.
        if method >= 4 and state.shape == (1, 6) and env.get_region(stick_pos) != env.INSIDE_HEMISPHERE:
            V_stick = 0.5 * torch.norm(torch.linalg.cross(stick_pos - cyl_center, cyl_axis)) ** 2
            V_stick = V_stick.view(1, 1)
            grad_V_stick = torch.linalg.cross(cyl_axis, torch.linalg.cross(stick_pos - cyl_center, cyl_axis)) * 0.9
            grad_V_stick = torch.cat([torch.zeros(1, 3).to(self.device), grad_V_stick], dim=1)
            V = torch.cat([V, V_stick], dim=0)
            grad_V = torch.cat([grad_V, grad_V_stick], dim=0)
            num_points += 1
            assert V.shape == (num_points, 1), f"Unexpected V shape: {V.shape}"
            assert grad_V.shape == (num_points, 6), f"Unexpected grad_V shape: {grad_V.shape}"    
        
        if V.shape == (0,):
            return u
        
        # Gradient of V times the dynamics
        LfV = grad_V @ fx.T
        LgV = grad_V @ gx.T
        
        assert LfV.shape == (num_points, 1)
        assert LgV.shape == (num_points, self.u_dim)
        
        decay_rate = 1.0
        G = LgV.to(self.device)
        h = -(LfV + decay_rate * V).to(self.device)
        P = torch.eye(self.u_dim).to(self.device)
        q = -u.T
        
        modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
        modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
        return modified_u
        
    @torch.no_grad()
    def hemipuncture_cbf_cylinder(
        self,
        u: torch.Tensor,
        env: Hemipuncture
    ) -> torch.Tensor:
        assert u.shape == (1, self.u_dim), "The 'u' tensor must have shape (1, 3)."
        
        psm_pos = env._get_robot_state(0)[:3]
        pitch_pos = np.array(get_link_pose(env.psm1.body, 4)[0])
        rc_pos = np.array(get_link_pose(env.psm1.body, 13)[0])
        stick_pos = pitch_pos + 0.1 * (rc_pos - pitch_pos)
        cyl_center, cyl_axis, _, cyl_radius = env.get_cylinder_prop()
        in_eff_region = (
            env.get_region(psm_pos) == env.INSIDE_CYLINDER or
            env.get_region(psm_pos) == env.INSIDE_CYLINDER or
            env.get_region(psm_pos) == env.INSIDE_CYLINDER
        )
        
        # Convert to torch tensor
        psm_pos = torch.from_numpy(psm_pos).float().unsqueeze(0).to(self.device)
        pitch_pos = torch.from_numpy(pitch_pos).float().unsqueeze(0).to(self.device)
        cyl_center = torch.from_numpy(cyl_center).float().unsqueeze(0).to(self.device)
        cyl_axis = torch.from_numpy(-cyl_axis).float().unsqueeze(0).to(self.device)
        assert psm_pos.shape == (1, 3), f"Unexpected psm_pos shape: {psm_pos.shape}"
        assert pitch_pos.shape == (1, 3), f"Unexpected pitch_pos shape: {pitch_pos.shape}"
        assert cyl_center.shape == (1, 3), f"Unexpected cyl_center shape: {cyl_center.shape}"
        assert cyl_axis.shape == (1, 3), f"Unexpected cyl_axis shape: {cyl_axis.shape}"
        
        # Obtain the dynamics
        state = torch.cat([psm_pos, pitch_pos], dim=1)
        net_out = self.net(state)
        assert net_out.shape == (1, self.x_dim + self.x_dim * self.u_dim), f"Unexpected network output shape: {net_out.shape}"
        fx = net_out[:, :self.x_dim]
        gx = net_out[:, self.x_dim:]
        gx = torch.reshape(gx, (self.u_dim, self.x_dim))
        
        if in_eff_region:
            # psm_pos
            h1 = cyl_radius ** 2 - torch.norm(torch.linalg.cross(psm_pos - cyl_center, cyl_axis)) ** 2
            h1 = h1.view(1, 1)
            grad_h1 = -2 * torch.linalg.cross(cyl_axis, torch.linalg.cross(psm_pos - cyl_radius, cyl_axis))
            grad_h1 = torch.cat([grad_h1, torch.zeros(1, 3).to(self.device)], dim=1)
            assert grad_h1.shape == (1, 6), f"Unexpected grad_h1 shape: {grad_h1.shape}"
            
            # pitch_pos
            h2 = cyl_radius ** 2 - torch.norm(torch.linalg.cross(pitch_pos - cyl_center, cyl_axis)) ** 2
            h2 = h2.view(1, 1)
            grad_h2 = -2 * torch.linalg.cross(cyl_axis, torch.linalg.cross(pitch_pos - cyl_radius, cyl_axis))
            grad_h2 = torch.cat([torch.zeros(1, 3).to(self.device), grad_h2], dim=1)
            assert grad_h2.shape == (1, 6), f"Unexpected grad_h2 shape: {grad_h2.shape}"
            
            h = torch.cat([h1, h2], dim=0)
            grad_h = torch.cat([grad_h1, grad_h2], dim=0)
            assert h.shape == (2, 1), f"Unexpected h shape: {h.shape}"
            assert grad_h.shape == (2, 6), f"Unexpected grad_h shape: {grad_h.shape}"
            
            Lfh = grad_h @ fx.T
            Lgh = grad_h @ gx.T
            
            gamma = 1
            G = -Lgh.to(self.device)
            h = (Lfh + gamma * h).to(self.device)
            P = torch.eye(self.u_dim).to(self.device)
            q = -u.T

            modified_u = cvx_solver(P.double(), q.double(), G.double(), h.double())
            modified_u = torch.from_numpy(modified_u).float().to(self.device).reshape(1, -1)
            
            if (modified_u == u).all():
                print("Action is unchanged!")
            else:
                print("Action is modified")
                print("Original:", u)
                print("Modified:", modified_u)
                
                
            return modified_u
        else:
            return u
        