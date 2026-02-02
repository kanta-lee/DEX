"""
Data Processing for Surgical Robot Demonstrations

This script processes surgical robot demonstration data for ODE model training.
It extracts Cartesian position data (x, y, z coordinates) from observations and actions.

Key functionalities:
- Load and preprocess dataset from specified surgical task
- Save 3D Cartesian positions (x, y, z)                 -> 'obs_pos.npy'
- Save 3D Cartesian actions (dx, dy, dz)                -> 'acs_pos.npy'
- Save Euler orientation (roll, pitch, yaw) + jaw angle -> 'obs_orn.npy'
- Save d_yaw, jaw status (open or close)                -> 'acs_orn.npy'

Supported tasks:
- NeedlePick-v1, NeedlePick-v2
- GauzeRetrieve-v1, GauzeRetrieve-v2

NOTE: This code only supports tasks that use a single PSM (Patient Side Manipulator).
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

TASKS = [
    'NeedlePick-v0',
    'NeedlePick-v1', 
    'NeedlePick-v2', 
    'GauzeRetrieve-v1', 
    'GauzeRetrieve-v2',
    'NeedleReach-v0',
    'PegTransfer-v0'
]

parser = argparse.ArgumentParser('Training Data Generation for Neural ODE')
parser.add_argument(
    '--task',
    type=str,
    choices=TASKS,
    required=True
)
args = parser.parse_args()

# Create directory to store processed data
if not os.path.exists(f"data/{args.task}"):
    os.makedirs(f"data/{args.task}")
else:
    # Check if directory is empty
    if not any(os.scandir(f"data/{args.task}")):
        pass
    else:
        print(f"Directory {args.task} already exists and is not empty.")
        exit(0)

# Load raw demonstration data
data = np.load(
    f'../SurRoL/surrol/data/demo/data_{args.task}_random_100.npz', 
    allow_pickle=True
)
obs_data = data['obs']  # materialize to avoid multi-threaded zip reads
acs_pos = data['acs'][:, :, 0:3]       # [demo, timestep, [dx, dy, dz]]
acs_orn = data['acs'][:, :, 3:]  # [demo, timestep, [d_yaw, jaw]]
num_demo, num_timestep = data['obs'].shape
data.close()

# Determine task domain
if args.task not in TASKS:
    print("Unsupported task. Please choose from the following:")
    print(TASKS)
    exit(0)


'''
For ['NeedlePick-v0', 'NeedlePick-v1', 'NeedlePick-v2']:
Observation len = 33
obs[0:7]: PSM's pose (3 position, 3 euler angle, 1 jaw status)
obs[7:10]: Object's position (Needle's base link position) (I don't think it is useful because it is not exactly on the needle)
obs[10:13]: Object's relative position (obs[7:10] - obs[0:3])
obs[13:16]: Position of center of the needle (the usual pick up place)
obs[16:19]: Orientation of center of the needle (Euler angles)
obs[19:22]: Position of left end of the needle
obs[22:25]: Orientation of left end of the needle (Euler angles)
obs[25:28]: Position of right end of the needle
obs[28:31]: Orientation of right end of the needle (Euler angles)
'''

# Initialize arrays for single PSM tasks
obs_pos = np.zeros((num_demo, num_timestep, 3))  # [demo, timestep, xyz]
obs_orn = np.zeros((num_demo, num_timestep, 4))  # [demo, timestep, [roll, pitch, yaw, jaw_angle]]

obj_pos = np.zeros((num_demo, num_timestep, 3))  # [demo, timestep, leftxyz]
obj_orn = np.zeros((num_demo, num_timestep, 3))  # [demo, timestep, leftori]

# Extract data from observations using threads for speed

def _process_demo(demo_idx: int):
    demo_obs_pos = np.zeros((num_timestep, 3))
    demo_obs_orn = np.zeros((num_timestep, 4))
    demo_obj_pos = np.zeros((num_timestep, 3))
    demo_obj_orn = np.zeros((num_timestep, 3))

    for timestep_idx in range(num_timestep):
        obs = obs_data[demo_idx][timestep_idx]['observation']
        demo_obs_pos[timestep_idx, :] = obs[0:3]
        demo_obs_orn[timestep_idx, :] = obs[3:7]

        if args.task in ['NeedlePick-v0', 'NeedlePick-v1', 'NeedlePick-v2']:
            demo_obj_pos[timestep_idx, :] = obs[19:22]
            demo_obj_orn[timestep_idx, :] = obs[22:25]

    return demo_idx, demo_obs_pos, demo_obs_orn, demo_obj_pos, demo_obj_orn


with ThreadPoolExecutor() as executor:
    futures = {executor.submit(_process_demo, demo_idx): demo_idx for demo_idx in range(num_demo)}
    for future in tqdm(as_completed(futures), total=num_demo, desc='demos', unit='demo'):
        demo_idx, demo_obs_pos, demo_obs_orn, demo_obj_pos, demo_obj_orn = future.result()
        obs_pos[demo_idx] = demo_obs_pos
        obs_orn[demo_idx] = demo_obs_orn
        obj_pos[demo_idx] = demo_obj_pos
        obj_orn[demo_idx] = demo_obj_orn

# Save processed data to files
np.save(f'data/{args.task}/obs_pos.npy', obs_pos)
np.save(f'data/{args.task}/acs_pos.npy', acs_pos)
np.save(f'data/{args.task}/obs_orn.npy', obs_orn)
np.save(f'data/{args.task}/acs_orn.npy', acs_orn)
np.save(f'data/{args.task}/obj_pos.npy', obj_pos)
np.save(f'data/{args.task}/obj_orn.npy', obj_orn)

print(f"Data processing completed for {args.task}")
