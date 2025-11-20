import torch
import torch.nn as nn
from typing import List
import glob
from pathlib import Path

class NeuralODE(nn.Module):
    """
    Defines the derivative function of the Neural ODE, modeled as a
    control-affine system: dx/dt = f(x) + g(x)u.
    
    The neural network approximates both f(x) and g(x).
    """

    def __init__(self, layer_dims: List[int]):
        """
        Initializes the NeuralODE module.
        
        Args:
            layer_dims: A list of integers defining the network layers.
                        Example: [x_dim, 64, x_dim + x_dim * u_dim]
        """
        super(NeuralODE, self).__init__()

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least an input and output size.")
            
        self.x_dim = layer_dims[0]
        
        # Calculate u_dim from the output layer dimension
        # output_dim = x_dim (for f(x)) + x_dim * u_dim (for g(x))
        output_dim = layer_dims[-1]
        if (output_dim - self.x_dim) % self.x_dim != 0:
            raise ValueError("Output layer dimension is not consistent with x_dim.")
        self.u_dim = (output_dim - self.x_dim) // self.x_dim

        self.net = self.build_mlp(layer_dims, activation='gelu')

        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        # 'self.u' is stateful and must be set externally before calling forward.
        # This is a common pattern when using torchdiffeq.odeint with
        # time-varying controls.
        self.u: torch.Tensor = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivative dx/dt at time t with state x and control u.
        
        Args:
            t: Current time (tensor). Required by odeint, but not used here.
            x: Current state (tensor).
        
        Returns:
            The derivative dx/dt (tensor).
        """
        if self.u is None:
            raise RuntimeError("self.u has not been set. Set func.u = ... before calling odeint.")
            
        net_out = self.net(x)

        # f(x) is the first part of the output
        fx = net_out[..., :self.x_dim] 
        
        # g(x) (flattened) is the second part
        gx_flat = net_out[..., self.x_dim:]

        if self.training:
            # --- Batched Case (Training) ---
            # x shape: [B, 1, X_DIM] (e.g., [20, 1, 3])
            # u shape: [B, 1, U_DIM] (e.g., [20, 1, 3])

            # Reshape g(x) from [B, 1, U*X] to [B, U_DIM, X_DIM]
            gx = gx_flat.reshape(x.shape[0], self.u_dim, self.x_dim)
            
            # Batched Matmul:
            #   self.u   @   gx
            # [20, 1, 3] @ [20, 3, 3]  -> PyTorch broadcasts this to
            # [20] batch dim, with [1, 3] @ [3, 3] matrix op
            # Result is [20, 1, 3]
        else:
            # --- Unbatched Case (Evaluation) ---
            # x shape: [1, X_DIM] (e.g., [1, 3])
            # u shape: [1, U_DIM] (e.g., [1, 3])
            # Reshape g(x) from [1, U*X] to [U_DIM, X_DIM]
            gx = gx_flat.reshape(self.u_dim, self.x_dim)
            
            # Unbatched Matmul:
            # self.u @ gx
            # [1, 3] @ [3, 3]
            # Result is [1, 3]

        return fx + self.u @ gx

    def build_mlp(
        self,
        filters: List[int],
        no_act_last_layer: bool = True,
        activation: str = 'gelu'
    ) -> nn.Sequential:
        """
        Builds a multi-layer perceptron (MLP).
        
        Args:
            filters: List of layer dimensions.
            no_act_last_layer: If True, the last layer has no activation.
            activation: Name of the activation function to use.
            
        Returns:
            A nn.Sequential module.
        """
        act_fn_map = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
        }
        
        if activation not in act_fn_map:
            raise NotImplementedError(
                f'Not supported activation function {activation}')
        
        act_fn = act_fn_map[activation]
        
        modules = nn.ModuleList()
        for i in range(len(filters) - 1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters) - 2):
                modules.append(act_fn)

        return nn.Sequential(*modules)

    def load_latest_weight(self, task: str):
        """Load latest weight using absolute path"""
        weights_dir = Path(__file__).parent / "weights" / task
        
        pattern = str(weights_dir / "model_iter_*.pth")
        weight_files = glob.glob(pattern)
        
        if not weight_files:
            raise FileNotFoundError(f"No weight files found: {pattern}")
        
        latest_weight = max(weight_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"Loading latest weights: {latest_weight}")
        
        self.load_state_dict(torch.load(latest_weight))
        return latest_weight