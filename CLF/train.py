import torch
import torch.optim as optim
from torchdiffeq import odeint

import numpy as np
import os
import argparse
import time
from clf import CLF

torch.autograd.set_detect_anomaly(True)

tasks = ['NeedlePick-v0', 'GauzeRetrieve-v0',
         'NeedleReach-v0', 'PegTransfer-v0', 'NeedleRegrasp-v0',]

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str,
                    choices=['dopri8', 'adams'], default='dopri8')  # dopri5
parser.add_argument('--activation', type=str,
                    choices=['gelu', 'silu', 'tanh'], default='gelu')
parser.add_argument('--task', type=str, choices=tasks, default='NeedleRegrasp-v0')
# length of each trajectory
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=200)  # 2000
parser.add_argument('--test_freq', type=int, default=20)  # 2 20
parser.add_argument('--gpu', type=int, default=0)


args = parser.parse_args()

# Setting up device used for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

# Just a file number
exp_id = '1'

# Assume that each time step is 0.1
# Since the length of each trajectory is 50
# So we need a time vector that has size of 50 with each element differs by 0.1
t = torch.linspace(0., 4.9, args.data_size).to(device)

# Load dataset
obs = np.load(f'../Data/{args.task}/obs_orn.npy')  # [100, 51, 4]
if args.task != 'NeedleRegrasp-v0':
    obs = obs[:, :, 0:3]  # Ignore jaw angle
obs = torch.tensor(obs).float()  # [100, 51, 3]
# obs[3:6] = orientation in Euler
# obs[6] = jaw angle

acs = np.load(f'../Data/{args.task}/acs_orn.npy')  # [100, 50 ,2]
# acs[3] = d_yaw / d_pitch
# acs[4] = jaw open > 0, otherwise close

# NOTE: The below adjustment is for single PSM ONLY.
#       Psm's self.ACTION_MODE = 'yaw'
#       acs[3] *= np.deg2rad(30) # yaw, limit maximum change in rotation
#       Ignoring acs[4] for now
#       https://github.com/med-air/SurRoL/blob/main/surrol/tasks/psm_env.py#L196
if args.task == 'NeedleRegrasp-v0':
    acs *= np.deg2rad(15)
    assert acs.shape == (100, 50, 2)
else:
    acs = acs[:, :, 0] * np.deg2rad(30)  # [100, 50, 1]
    acs = acs.reshape((100, 50, 1))
acs = torch.tensor(acs).float()

# \dot{x} = f(x) + u * g(x)
# x = [1, 3] (obs)
# u = [1, 1] (acs)
# f = [1, 3] (net)
# g = [1, 3] (net)

# Training data
x_train = obs.unsqueeze(2).to(device)  # [100, 51, 1, 3]
u_train = acs.unsqueeze(2).to(device)  # [100, 50, 1, 1]

# Testing data
x_test = x_train[-1, :, :, :]  # [51, 1, 3]
u_test = u_train[-1, :, :, :]  # [50, 1, 1]

# Initial condition for testing
x_test0 = x_train[-1, 0, :, :]  # [1, 3]
u_test0 = u_train[-1, 0, :, :]  # [1, 1]


def get_batch(bitr):  # get training data from bitr-th trajectory
    # u = [ time, 1, actions ]
    u = u_train[bitr, :, :, :]  # [50, 1, 1]
    x = x_train[bitr, :, :, :]  # [51, 1, 3]

    # data_size  = length of traj = 50
    # batch_time = 10
    # batch_size = 20
    # np.arange(50-10) = [0, 1, 2, 3, ..., 39]
    # bb = array of size 19 with random numbers from [0, 39]
    # bb.shape = [19]
    bb = torch.from_numpy(
        np.random.choice(
            np.arange(args.data_size - args.batch_time, dtype=np.int64),
            args.batch_size-1,
            replace=False
        )
    )
    aa = torch.tensor([0])          # augment 0, aa.shape = [1]

    # Concat 0 to the end of bb
    s = torch.cat([bb, aa], dim=0)  # s.shape = [20]
    batch_u0 = u[s]  # (M, D) # [20, 1, 1]
    batch_x0 = x[s]  # [20, 1, 3]
    batch_t = t[:args.batch_time]  # (T)

    # u[s].shape = [20, 1, 1]
    batch_u = torch.stack(
        [u[s + i] for i in range(args.batch_time)], dim=0
    )  # (T, M, D) # [10, 20, 1, 1]
    # batch_u[time] returns actions of trajectory bitr starting from time "s" to time "s + 10"
    # u[s] returns the actions at time s, where s contains multiple time
    # u[s + i] returns the actions at time s + i
    # batch_u[0][start] returns actions at time start
    # batch_u[i][start] returns actions at time start + i

    batch_x = torch.stack(
        [x[s + i] for i in range(args.batch_time)], dim=0
    )  # [10, 20, 1, 3]

    return batch_u0.to(device), batch_x0.to(device), batch_t.to(device), batch_u.to(device), batch_x.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        print("Directory existed! Please recheck!")
        print("If you want to retrain, please increment exp_id variable")
        exit(0)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    test_count = 0

    # Create directory to store trained weights
    saved_folder = f'saved_model/{args.task}/{exp_id}'
    makedirs(saved_folder)

    # Set up the dimension of the network
    x_dim = x_train.shape[-1]
    u_dim = u_train.shape[-1]
    fc_param = [x_dim, 64, x_dim + x_dim * u_dim]

    # Initialize neural ODE
    func = CLF(fc_param).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # print(func)

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    end = time.time()

    test_loss = torch.tensor([0]).to(device)
    tt = torch.tensor([0.0, 0.1]).to(device)

    # Training Loop
    for iter in range(1, args.niters + 1):
        # Loop over each trajectory
        for traj in range(x_train.shape[0]):

            func.train()
            optimizer.zero_grad()
            batch_u0, batch_x0, batch_t, \
                batch_u, batch_x = get_batch(traj)

            # Setup input for network
            x0 = batch_x0             # [20, 1, 3]
            pred_x = x0.unsqueeze(0)  # [1, 20, 1, 3]

            # Loop over each time step in a time batch
            for i in range(args.batch_time-1):

                # Store this to use u in forward() later
                batch_u_i = batch_u[i, :, :, :]
                func.u = batch_u_i

                # pred: [ x0, x0 at next timestep ]
                # returns next observation in the last index
                # (depend on what tt is)
                pred = odeint(func, x0, tt).to(device)  # [2, 20, 1, 3]

                # Concat predicted next state into pred_x
                pred_x = torch.cat(
                    [pred_x, pred[-1, :, :, :].unsqueeze(0)], dim=0)

                # Update input to the network
                x0 = pred[-1, :, :, :]

            # Compute loss [10, 20, 1, 3]
            loss = torch.mean(torch.abs(pred_x - batch_x))

            # Display loss
            print('iteration: ', iter,
                  '| traj: ', traj,
                  '| train loss: ', loss.item())

            loss.backward()
            optimizer.step()

            # No idea what this is for
            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

        if iter % args.test_freq == 0:
            with torch.no_grad():
                # Set input for network
                x0 = x_test0
                pred_x = x0.unsqueeze(0)
                func.eval()

                # Test over the whole test trajectory
                # Predict every timestep
                # Compute cost with real data
                for i in range(args.data_size):
                    # Setup u for forward()
                    # u_test[i,:,:] is the actions of test trajectory that at time i
                    u_test_i = u_test[i, :, :]
                    func.u = u_test_i

                    pred = odeint(func, x0, tt)
                    pred_x = torch.cat(
                        [pred_x, pred[-1, :, :].unsqueeze(0)], dim=0)

                    # Update input
                    x0 = pred[-1, :, :]

                # [51, 1, 3]
                test_loss = torch.mean(torch.abs(pred_x - x_test))

                # Display info
                print('Iter {:04d} | Test Loss {:.6f}'.format(
                    iter, test_loss.item()))

                test_count += 1

            torch.save(func.state_dict(),
                       f"{saved_folder}/CLF{format(test_count, '02d')}.pth")
