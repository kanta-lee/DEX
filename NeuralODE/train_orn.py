import argparse
import os
import torch
import torch.optim as optim
import numpy as np
from torch import Tensor
from typing import Tuple
from torchdiffeq import odeint

from node import NeuralODE

# Enable anomaly detection for debugging gradients
torch.autograd.set_detect_anomaly(True)

TASKS = [
    'NeedlePick-v1',
    'NeedlePick-v2',
    'GauzeRetrieve-v1',
    'GauzeRetrieve-v2'
]

def setup_argparser() -> argparse.ArgumentParser:
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser('Neural ODE Training Script')
    parser.add_argument('--method', type=str, choices=['dopri8', 'adams'], default='dopri8')
    parser.add_argument('--activation', type=str, choices=['gelu', 'silu', 'tanh'], default='gelu')
    parser.add_argument('--task', type=str, choices=TASKS, required=True)
    parser.add_argument('--data_size', type=int, default=100, help="Length of each trajectory's action sequence")
    parser.add_argument('--batch_time', type=int, default=10, help="Length of time steps in a batch")
    parser.add_argument('--batch_size', type=int, default=20, help="Number of trajectory segments in a batch")
    parser.add_argument('--niters', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--test_freq', type=int, default=20, help="Frequency to run evaluation")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    
    return parser

def load_data(task_name: str, device: torch.device) -> Tuple:
    """Loads and pre-processes training and testing data."""
    print(f"Loading data for task: {task_name}")
    
    try:
        obs = np.load(f'data/{task_name}/obs_orn.npy') 
        acs = np.load(f'data/{task_name}/acs_orn.npy')
    except FileNotFoundError:
        print(f"Error: Data files not found in 'data/{task_name}/'.")
        print("Please ensure the data is correctly placed.")
        exit(1)

    # obs_orn: [num_demo, num_timestep, 4] - [roll, pitch, yaw, jaw_angle]
    # Excluding jaw_angle as it's not part of the control space
    obs = obs[:, :, 0:3]

    # Using only d_yaw (scaled by 30 degrees to radians) as control input
    # jaw_status (0.5: open, -0.5: closed) is excluded as it's a discrete action
    acs = acs[:, :, [0]] * np.deg2rad(30)

    # Convert to torch tensor
    obs = torch.from_numpy(obs).float().to(device)
    acs = torch.from_numpy(acs).float().to(device)

    # Add a singleton dimension for 'channel'
    x_all = obs.unsqueeze(2)
    u_all = acs.unsqueeze(2)

    # Use the last trajectory for testing (held-out set)
    x_test = x_all[-1, :, :, :] # Shape [data_size + 1, 1, x_dim]
    u_test = u_all[-1, :, :, :] # Shape [data_size, 1, u_dim]
    
    # Use all but the last trajectory for training
    x_train = x_all[:-1, :, :, :]
    u_train = u_all[:-1, :, :, :]

    # Initial condition for testing
    x_test0 = x_test[0, :, :] # Shape [1, x_dim]
    
    print(f"Training data: x shape {x_train.shape}, u shape {u_train.shape}")
    print(f"Test data: x shape {x_test.shape}, u shape {u_test.shape}")

    return x_train, u_train, x_test, u_test, x_test0

def get_batch(
    bitr: int, 
    x_train: Tensor, 
    u_train: Tensor, 
    args: argparse.Namespace
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Get a batch of data from the bitr-th trajectory.
    
    This function uses efficient tensor indexing.
    """
    # Get the specific trajectory
    u = u_train[bitr, :, :, :]  # [data_size, 1, u_dim]
    x = x_train[bitr, :, :, :]  # [data_size + 1, 1, x_dim]

    # Select random starting time indices
    s_indices = torch.from_numpy(
        np.random.choice(
            np.arange(args.data_size - args.batch_time, dtype=np.int64),
            args.batch_size - 1, # -1 because we always add index 0
            replace=False
        )
    ).to(u.device)
    
    # Always include the start of the trajectory (index 0)
    s = torch.cat([torch.tensor([0]).to(u.device), s_indices], dim=0)  # [batch_size]
    
    batch_x0 = x[s]  # [batch_size, 1, x_dim]

    # Create a [batch_time, batch_size] tensor of indices
    # This efficiently selects 'batch_time' steps for each 'batch_size' start point
    time_indices = torch.arange(args.batch_time).to(u.device).unsqueeze(1) # [batch_time, 1]
    batch_indices = s.unsqueeze(0) # [1, batch_size]
    
    # Broadcasting creates [batch_time, batch_size]
    full_indices = time_indices + batch_indices 

    # Gather all data at once
    # u's indices are [0, data_size-1], x's are [0, data_size]
    batch_u = u[full_indices]  # [batch_time, batch_size, 1, u_dim]
    batch_x = x[full_indices]  # [batch_time, batch_size, 1, x_dim]

    return batch_x0, batch_u, batch_x


def run_evaluation(
    func: NeuralODE, 
    x_test: Tensor, 
    u_test: Tensor, 
    x_test0: Tensor, 
    t_step_vec: Tensor,
    data_size: int
):
    """Runs the model over the full test trajectory."""
    
    print("Running evaluation...")
    with torch.no_grad():
        x0 = x_test0             # Shape [1, x_dim]
        pred_x_test = x0.unsqueeze(0) # Shape [1, 1, x_dim]
        func.eval()

        # Test over the whole test trajectory
        for i in range(data_size):
            func.u = u_test[i, :, :] # Shape [1, u_dim]
            
            # Integrate one time step
            pred = odeint(func, x0, t_step_vec, method='dopri8') # [2, 1, x_dim]
            
            # Get the predicted state at t=0.1
            x_next = pred[-1, :, :] # Shape [1, x_dim]
            
            # Concat predicted next state
            pred_x_test = torch.cat(
                [pred_x_test, x_next.unsqueeze(0)], dim=0)

            # Update input for next step
            x0 = x_next

        # Compute loss against the full ground truth
        # x_test shape is [data_size + 1, 1, x_dim]
        test_loss = torch.mean(torch.abs(pred_x_test - x_test))
        print(f"Evaluation Loss: {test_loss.item():.6f}")

def train(args: argparse.Namespace):
    """Main training loop."""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directory to store trained weights
    saved_folder = f'weights/{args.task}'
    os.makedirs(saved_folder, exist_ok=True) # No need to check if empty

    # Load data
    x_train, u_train, x_test, u_test, x_test0 = load_data(
        args.task, device
    )
    
    num_trajs = x_train.shape[0]
    x_dim = x_train.shape[-1]
    u_dim = u_train.shape[-1]
    
    # Set up the dimension of the network
    # Input: x_dim
    # Output: f(x) [x_dim] + g(x) [x_dim * u_dim]
    layer_dims = [x_dim, 64, x_dim + x_dim * u_dim]

    # Initialize neural ODE
    func = NeuralODE(layer_dims).to(device)
    
    # Optimizer was missing from original file
    optimizer = optim.RMSprop(func.parameters(), lr=args.lr)

    # Time vector for single-step integration
    t_step_vec = torch.tensor([0.0, 0.1]).to(device)

    # Training Loop
    for itr in range(1, args.niters + 1):
        # Loop over each trajectory in the training set
        for traj_idx in range(num_trajs):

            func.train()
            optimizer.zero_grad()
            
            batch_x0, batch_u, batch_x = get_batch(traj_idx, x_train, u_train, args)

            # --- ODE Rollout ---
            x0 = batch_x0             # [batch_size, 1, x_dim]
            pred_x_rollout = x0.unsqueeze(0)  # [1, batch_size, 1, x_dim]

            # Loop over each time step in the batch
            for i in range(args.batch_time - 1):
                # Set the control input for this time step
                func.u = batch_u[i, :, :, :] # Shape [batch_size, 1, u_dim]
                
                # Integrate one step forward
                pred = odeint(func, x0, t_step_vec, method=args.method)  # [2, batch_size, 1, x_dim]

                # Wrap angle to be between [-np.pi, np.pi]
                pred = torch.remainder(pred + np.pi, 2 * np.pi) - np.pi
                
                # Get the predicted state at t=0.1
                x_next = pred[-1, :, :, :]

                # Concat predicted next state
                pred_x_rollout = torch.cat(
                    [pred_x_rollout, x_next.unsqueeze(0)], dim=0)

                # Update input for the next step
                x0 = x_next
            # --- End ODE Rollout ---

            # Compute loss
            loss = torch.mean(torch.abs(pred_x_rollout - batch_x))

            loss.backward()
            optimizer.step()
            
            if (traj_idx + 1) % 10 == 0:
                print(f'Iter: {itr}/{args.niters} | Traj: {traj_idx+1}/{num_trajs} | Train Loss: {loss.item():.6f}')


        # --- Evaluation ---
        if itr % args.test_freq == 0:
            run_evaluation(
                func, x_test, u_test, x_test0, t_step_vec, args.data_size
            )
            # Save weights
            save_path = f'{saved_folder}/model_orn_iter_{itr}.pth'
            torch.save(func.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = setup_argparser()
    args = parser.parse_args()
    train(args)