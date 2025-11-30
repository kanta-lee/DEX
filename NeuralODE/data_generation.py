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

import numpy as np
import argparse
import os

TASKS = [
    'NeedlePick-v1', 
    'NeedlePick-v2', 
    'GauzeRetrieve-v1', 
    'GauzeRetrieve-v2'
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

# Determine task domain
if args.task not in TASKS:
    print("Unsupported task. Please choose from the following:")
    print(TASKS)
    exit(0)

num_demo, num_timestep = data['obs'].shape

# Initialize arrays for single PSM tasks
obs_pos = np.zeros((num_demo, num_timestep, 3))  # [demo, timestep, xyz]
acs_pos = data['acs'][:, :, 0:3]       # [demo, timestep, [dx, dy, dz]]
obs_orn = np.zeros((num_demo, num_timestep, 4))  # [demo, timestep, [roll, pitch, yaw, jaw_angle]]
acs_orn = data['acs'][:, :, 3:]  # [demo, timestep, [d_yaw, jaw]]

# Extract position data from observations
for demo_idx in range(num_demo):
    for timestep_idx in range(num_timestep):
        # Extract x, y, z coordinates from observation
        obs_pos[demo_idx, timestep_idx, :] = \
            data['obs'][demo_idx][timestep_idx]['observation'][0:3]
        
        # Extract w, x, y, z coordinates from observation
        obs_orn[demo_idx, timestep_idx, :] = \
            data['obs'][demo_idx][timestep_idx]['observation'][3:7]

# Save processed data to files
np.save(f'data/{args.task}/obs_pos.npy', obs_pos)
np.save(f'data/{args.task}/acs_pos.npy', acs_pos)
np.save(f'data/{args.task}/obs_orn.npy', obs_orn)
np.save(f'data/{args.task}/acs_orn.npy', acs_orn)

print(f"Data processing completed for {args.task}")
