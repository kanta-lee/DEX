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

    # TODO: Implement the CLF for the task.
    #       Function name can be set as the task name.
    #       For example,
    #       
    #       def needle_pick_spiral(self, u, env):
    #           ...
    #
    #       For the implementation code, see the CBF class.
    #
    #       To utilize PositionCLF, you can directly use the trained
    #       Neural ODE network. However, there are weights for NeedlePick-v1,
    #       NeedlePick-v2, GauzeRetrieve-v1, and GauzeRetrieve-v2 only.
    #      
    #       If you want to use PositionCLF for other tasks, you need to
    #       first train the weights for the task in NeuralODE directory.
    #       
    #       When use, you can first initialize the PositionCLF object in
    #       samplers.py __init__(). Then, you can use it in the same way as
    #       CBF by calling the function in sample_episode().