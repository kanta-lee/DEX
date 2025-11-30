# Neural ODE for Robot Control

This directory implements a Neural Ordinary Differential Equation (Neural ODE) for modeling and controlling robotic systems. The model learns the dynamics of a robotic system in a control-affine form: 

$$ \dot{x} = f(x) + g(x)u $$

where $x$ is the state and $u$ is the control input.

## Features

- Implements a Neural ODE that models system dynamics using a multi-layer perceptron (MLP)
- Supports different activation functions (GELU, SiLU, Tanh)
- Includes training and evaluation scripts
- Handles batched training data for efficient learning
- Saves model checkpoints during training

<!-- ## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd DEX/NeuralODE
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchdiffeq numpy
   ``` -->

## Usage

### Training

To train the Neural ODE model on a specific task:

```bash
python train.py \
    --task NeedlePick-v1 \
    --method dopri8 \
    --activation gelu \
    --data_size 100 \
    --batch_time 10 \
    --batch_size 20 \
    --niters 200 \
    --test_freq 20 \
    --lr 1e-3
```

#### Available Tasks
- NeedlePick-v1
- NeedlePick-v2
- GauzeRetrieve-v1
- GauzeRetrieve-v2

<!-- ### Training Scripts

There are two convenience scripts for training:

1. `train_all_task.sh`: Trains the model on all available tasks
2. `train_passive.sh`: Trains the model in passive mode (if implemented) -->

## Model Architecture

The Neural ODE consists of:
- An MLP that predicts both f(x) and g(x) in the control-affine system
- Configurable number of layers and hidden dimensions
- Support for different activation functions
- Batched and non-batched inference modes

## Data Generation

To generate training data from surgical robot demonstrations, use the `data_generation.py` script:

```bash
python data_generation.py --task [TASK_NAME]
```

### Available Tasks
- `NeedlePick-v1`
- `NeedlePick-v2`
- `GauzeRetrieve-v1`
- `GauzeRetrieve-v2`

### Output Files
For each task, the script generates the following files in the `data/{task_name}/` directory:
- `obs_pos.npy`: 3D Cartesian positions (x, y, z) of the robot end-effector
- `acs_pos.npy`: 3D Cartesian actions (dx, dy, dz)
- `obs_orn.npy`: Orientation data (quaternion w, x, y, z)
- `acs_orn.npy`: Orientation actions (yaw, jaw status)

### Data Format

The training data is organized as follows:
```
data/
  {task_name}/
    obs_pos.npy    # Observations/States [num_trajectories, traj_length+1, state_dim]
    acs_pos.npy    # Actions/Controls [num_trajectories, traj_length, action_dim]
    obs_orn.npy    # Orientation data [num_trajectories, traj_length+1, 4] (quaternion)
    acs_orn.npy    # Orientation actions [num_trajectories, traj_length, 2] (yaw, jaw)
```

### Dependencies
- NumPy
- SurRoL (for demonstration data)

## Saving and Loading Models

Trained models are automatically saved in the `weights/{task_name}/` directory. To load a pre-trained model:

```python
from node import NeuralODE

# Initialize model with the same architecture
model = NeuralODE([state_dim, 64, 64, state_dim + state_dim * action_dim])

# Load weights
model.load_latest_weight('NeedlePick-v1')
```

<!-- ## Dependencies

- Python 3.6+
- PyTorch
- torchdiffeq
- NumPy -->
