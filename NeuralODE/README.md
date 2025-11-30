# Neural ODE for Robot Control

This repository implements Neural Ordinary Differential Equations (Neural ODEs) for modeling and controlling robotic systems. The models learn the dynamics of robotic systems in a control-affine form:

$$\dot{x} = f(x) + g(x)u$$

where $x$ is the state and $u$ is the control input. We provide three training scripts for different aspects of robot control:

1. `train_pos.py`: Trains position dynamics only
2. `train_orn.py`: Trains orientation dynamics only
3. `train.py`: Trains combined position and orientation dynamics

## Features

- Implements Neural ODEs that model system dynamics using multi-layer perceptrons (MLPs)
- Supports different activation functions (GELU, SiLU, Tanh)
- Includes training and evaluation scripts for position, orientation, and combined dynamics
- Handles batched training data for efficient learning
- Saves model checkpoints during training
- Integrates with CBF (Control Barrier Functions) and CLF (Control Lyapunov Functions) for safe control

## Workflow

### 1. Data Generation

First, generate training data using the `data_generation.py` script:

```bash
python data_generation.py --task [TASK_NAME]
```

Available tasks:
- `NeedlePick-v1`
- `NeedlePick-v2`
- `GauzeRetrieve-v1`
- `GauzeRetrieve-v2`

This script generates the following files in `data/{task_name}/`:
- `obs_pos.npy`: 3D Cartesian positions (x, y, z)
- `acs_pos.npy`: 3D Cartesian actions (dx, dy, dz)
- `obs_orn.npy`: Orientation data (roll, pitch, yaw, jaw_angle)
- `acs_orn.npy`: Orientation actions (d_yaw, jaw_status)

### 2. Model Training

#### Position Dynamics Only

```bash
python train_pos.py --task [TASK_NAME] \
                   --method dopri8 \
                   --activation gelu \
                   --data_size 100 \
                   --batch_time 10 \
                   --batch_size 20 \
                   --niters 200 \
                   --test_freq 20 \
                   --lr 1e-3
```

#### Orientation Dynamics Only

```bash
python train_orn.py --task [TASK_NAME] \
                   --method dopri8 \
                   --activation gelu \
                   --data_size 100 \
                   --batch_time 10 \
                   --batch_size 20 \
                   --niters 200 \
                   --test_freq 20 \
                   --lr 1e-3
```

#### Combined Position and Orientation

```bash
python train.py --task [TASK_NAME] \
               --method dopri8 \
               --activation gelu \
               --data_size 100 \
               --batch_time 10 \
               --batch_size 20 \
               --niters 200 \
               --test_freq 20 \
               --lr 1e-3
```

### 3. Model Usage in Samplers

The trained Neural ODE models are used in the DEX framework for safe control. The `Sampler` class in `dex/modules/samplers.py` integrates the Neural ODE with CBF and CLF for safe trajectory generation.

Key components:

1. **Neural ODE Initialization**:
   ```python
   # Initialize Neural ODE
   self.node = NeuralODE([3, 64, 12]).to(self.device)
   self.node.load_latest_weight(self.cfg.task)
   self.node.eval()

   # NOTE: If you want to use the old weights that was trained using position data only,
   #       make sure you change layers' parameters to [3, 64, 64, 12]. I accidentally
   #       type one more 64. This will be removed in latest version for faster training.

   # NOTE: Make sure to check the implementation of load_latest_weight to call the
   #       desired weights.
   #
   #       Currently, the weight filename has changed for each type of training.
   #       train.py -> model_iter_100.pth
   #       train_pos.py -> model_pos_iter_100.pth
   #       train_orn.py -> model_orn_iter_100.pth
   ```

2. **CBF/CLF Integration**:
   The sampler uses the Neural ODE model to compute safe control actions using either CBF or CLF:
   ```python
   # For CBF
   if not is_train and self.cfg.use_dcbf:
       # CBF-based control logic
       modified_action = self.cbf.needle_pick_sphere(u, env)
       
   # For CLF
   if not is_train and self.cfg.use_dclf:
       # CLF-based trajectory tracking
       modified_action, p_ref = self.clf.traj_tracking(u, env)

   # NOTE: If you are using the combined NeuralODE, CBF and CLF code might need
   #       to be changed to handle additional data because the old code only
   #       handles position data.
   ```
   