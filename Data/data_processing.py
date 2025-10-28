"""
This file implements a data processing before input into ODE model.

Key functionalities:
- Load and preprocess the dataset from specified task.
- Save x, y, z position in Cartesian            ->   'obs_pos.npy'
- Save orientation in Euler and jaw angle       ->   'obs_orn.npy'
- Save position movement in Cartesian space     ->   'acs_pos.npy'
- Save orientation movement (d_yaw / d_pitch)   ->   'acs_orn.npy'
  and open status of jaw

Usage:
python data_preprocessing.py --task=NeedlePick-v0

NOTE: This code only supports tasks that only use a single PSM.
"""

import numpy as np
import argparse
import os

# # Check current working directory
# print(os.getcwd())

parser = argparse.ArgumentParser('Data Processing')
parser.add_argument(
    '--task',
    type=str,
    choices=['NeedlePick-v0', 'GauzeRetrieve-v0', 'GauzeRetrieve-v1',
             'NeedleReach-v0', 'PegTransfer-v0',
             'NeedleRegrasp-v0', 'NeedlePick-v1', 'Hemipuncture-v0'],
    default='NeedlePick-v0'
)
args = parser.parse_args()

# Create a directory to store data
if not os.path.exists(args.task):
    os.makedirs(args.task)
else:
    # If directory is empty, continue
    if not any(os.scandir(args.task)):
        pass
    else:
        print(f"Directory {args.task} already exists.")
        exit(0)

# Load raw export demonstration data
data = np.load(
    f'../SurRoL/surrol/data/demo/data_{args.task}_random_100.npz', allow_pickle=True)

# Specify type of the task
ECM = 0
SINGLE_PSM = 1
BI_PSM = 2
if args.task in ['NeedlePick-v0', 'NeedlePick-v1', 'GauzeRetrieve-v0', 'GauzeRetrieve-v1', 'NeedleReach-v0', 'PegTransfer-v0']:
    domain = SINGLE_PSM
elif args.task in ['NeedleRegrasp-v0']:
    domain = BI_PSM
else:
    domain = ECM

'''
>>> data.files
['acs', 'obs', 'info']
>>> data['obs'].shape
(100, 51)
>>> data['obs'][0][0]
{'observation': array([ 2.50000024e+00,  2.50000298e-01,  3.60099983e+00, -1.57077880e+00,
       -6.55750109e-06,  1.57090934e+00, -4.43718019e-09,  2.70893839e+00,
        1.00540941e-01,  3.41057810e+00,  2.08938156e-01, -1.49459357e-01,
       -1.90421737e-01,  2.69789934e+00,  1.35331601e-01,  3.41057682e+00,
       -1.67208651e-06,  3.32018364e-06, -1.26354618e+00]), 
 'achieved_goal': array([2.69789934, 0.1353316 , 3.41057682]), 
 'desired_goal': array([2.73656776, 0.03006118, 3.576     ])}
'''

''' BI-PSM
>>> obs = data['obs'][0][0]['observation']
>>> obs.shape
(35,)

obs[0:7]: PSM1's state (obs[0:3]: pose_world, obs[3:6]: Euler orn, obs[6]: jaw angle)
obs[7:14]: PSM2's state (obs[7:10]: pose_world, obs[10:13]: Euler orn, obs[13]: jaw angle)
obs[14:17]: Object position (Needle, Gauze, ...)
obs[17:20]: obs[14:17] - obs[0:3] (Object relative position from PSM1)
obs[20:23]: obs[14:17] - obs[7:10] (Object relative position from PSM2)
obs[23:26]: Waypoint Position of PSM1
obs[26:29]: Waypoint Orientation of PSM2
obs[29:32]: Waypoint Position of PSM2
obs[32:35]: Waypoint Orientation of PSM1
'''

# NOTE: Different tasks have different shape of observation.
#       This is differentiate by self.has_object.
#       Single PSM tasks with self.has_object=True have observation of shape 19
#       Single PSM tasks with self.has_object=False have observation of shape 7
#       Action shape remains the same for all single PSM tasks.

num_ep = data['obs'].shape[0]
assert num_ep == data['acs'].shape[0]

timestep_per_ep = data['obs'].shape[1]
assert timestep_per_ep == data['acs'].shape[1] + 1


if domain == SINGLE_PSM:
    # Initialize arrays
    obs_pos = np.zeros((num_ep, timestep_per_ep, 3))
    obs_orn = np.zeros((num_ep, timestep_per_ep, 4))  # jaw angle included
    acs_pos = data['acs'][:, :, 0:3]
    acs_orn = data['acs'][:, :, 3:5]  # d_yaw, jaw open / close
    
    # Get observation data
    for i in range(num_ep):
        print(f"Episode {i}")
        for j in range(timestep_per_ep):
            obs_pos[i, j, :] = data['obs'][i][j]['observation'][0:3]
            obs_orn[i, j, :] = data['obs'][i][j]['observation'][3:7]
elif domain == BI_PSM:
    obs_pos = np.zeros((num_ep, timestep_per_ep, 3+3))
    obs_orn = np.zeros((num_ep, timestep_per_ep, 3+3))  # jaw angle NOT included
    acs_pos = data['acs'][:, :, [0, 1, 2, 5, 6, 7]]
    acs_orn = data['acs'][:, :, [3, 8]]  # d_yaw

    for i in range(num_ep):
        print(f"Episode {i}")
        for j in range(timestep_per_ep):
            obs_pos[i, j, :] = data['obs'][i][j]['observation'][[0, 1, 2, 7, 8, 9]]
            obs_orn[i, j, :] = data['obs'][i][j]['observation'][[3, 4, 5, 10, 11, 12]]
elif args.task == "Hemipuncture-v0":
    # NOTE: Position of tool_pitch is included.
    obs_pos = np.zeros((num_ep, timestep_per_ep, 6))
    acs_pos = data['acs'][:, :, 0:3]
    
    # Get observation data
    for i in range(num_ep):
        print(f"Episode {i}")
        for j in range(timestep_per_ep):
            obs_pos[i, j, :] = data['obs'][i][j]['observation'][[0, 1, 2, 7, 8, 9]]


# Save to npy files
if args.task == "Hemipuncture-v0":
    np.save(f'{args.task}/obs_pos.npy', obs_pos)
    np.save(f'{args.task}/acs_pos.npy', acs_pos)
else:
    np.save(f'{args.task}/obs_pos.npy', obs_pos)
    np.save(f'{args.task}/obs_orn.npy', obs_orn)
    np.save(f'{args.task}/acs_pos.npy', acs_pos)
    np.save(f'{args.task}/acs_orn.npy', acs_orn)
