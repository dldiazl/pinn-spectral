"""Fully connected neural network used by the NAS and PINN stages.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import numpy as np
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Fixed seed used by the legacy implementation for deterministic training.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FullyConnectedNN(nn.Module):
    """Fully connected network trained with full-batch L-BFGS.

    Used both by the supervised NAS stage and, through architecture transfer,
    by the physics-informed training stage. The activation function is
    applied after every hidden layer and omitted after the output layer.
    """

    def __init__(self, layers, activation, max_epochs=1000):
        """Build the network layers, initialize weights, and set up the L-BFGS optimizer.

        :param layers: Layer widths, including the input and output layers.
        :param activation: Activation name, one of ``tanh``, ``relu``, ``sigmoid``.
        :param max_epochs: Maximum number of outer optimization epochs.
        """
        super(FullyConnectedNN, self).__init__()

        # Activation-function map.
        activation_map = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
        }

        # Create the neural-network layers.
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # Do not add an activation after the output layer.
                if activation not in activation_map:
                    raise ValueError(f"Activation function '{activation}' is not valid. Options: {list(activation_map.keys())}")
                self.layers.append(activation_map[activation])
        
        self.net = nn.Sequential(*self.layers)

        # Initialize weights with the legacy routine.
        self._initialize_weights()

        # L-BFGS optimizer initialization.
        self.optimizer = optim.LBFGS(self.parameters(), lr=1, max_iter=20, tolerance_grad=1e-9, tolerance_change=np.finfo(float).eps, line_search_fn="strong_wolfe", history_size=100)
        
        # DataFrame used to store the training history.
        self.history = pd.DataFrame(columns=["Epoch", "Loss", "LossIC", "LossBC", "LossPhysics", "Time_optimizer", "Time_cpu", "Flops" , "LR"])
        
        self.max_epochs = max_epochs
        
        self.current_epoch = 1
        
        self.loss = None
        self.lossBC = None
        self.lossIC = None
        self.lossPhysics = None
        
        
        
    def _initialize_weights(self):
        """Initialize layer weights with Xavier initialization and zero biases."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Legacy deterministic Xavier initialization.
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward propagation."""
        return self.net(x)
    
  
    def compute_gradients(self, X):
        """
        Compute explicit derivatives of the network output with respect to the input.
    
        :param X: Input tensor with shape (N, 2), where the first column is t and the second is x.
        :return: Network output y, derivative u_t, derivative u_x, and derivative u_xx.
        """
        X = X.clone()
        X.requires_grad = True  # Allow PyTorch to compute gradients.
        y = self.forward(X)  # Forward propagation.
    
        # Compute first-order gradients.
        dy_dX = torch.autograd.grad(outputs=y, inputs=X, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        u_t = dy_dX[:, 0]  # Derivative with respect to t.
        u_x = dy_dX[:, 1]  # Derivative with respect to x.
    
        # Compute second-order gradients, u_xx.
        dy_dXX = torch.autograd.grad(outputs=u_x, inputs=X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = dy_dXX[:, 1]  # Second derivative with respect to x.
        
        return y, u_t, u_x, u_xx
    
    def loss_Physics(self, XPhysics, loss_physics_function):
        """
        Compute the domain loss using the supplied differential-equation residual.

        :param X_c: Interior or collocation points where the differential equation is evaluated.
        :return: Domain loss.
        """
        yPredPhysics, u_tPhysics, u_xPhysics, u_xxPhysics = self.compute_gradients(XPhysics)  # Required derivatives.
        
        # Use the custom differential-equation function to compute the physics loss.
        lossPhysics = loss_physics_function(XPhysics, yPredPhysics, u_tPhysics, u_xPhysics, u_xxPhysics)
        return lossPhysics


    def fit(self, XBC, yBC, XIC=None, yIC=None, XPhysics=None, loss_physics_function=None, patience = 20, updateLR = False):
        """Train with L-BFGS and the legacy closure definition."""
        
        # self.optimizer.zero_grad()
        
        # Closure definition.
        def closure():
            """Evaluate the combined loss and its gradient for one L-BFGS step."""
            self.optimizer.zero_grad()
            
            #with torch.no_grad():
            # Compute the network output.
            yPredBC = self.forward(XBC)
                
            # Compute the loss.
            lossBC = nn.MSELoss()(yPredBC, yBC)
                
            if XIC is None:
                lossIC = 0.0
            else:
                yPredIC = self.forward(XIC)
                lossIC = nn.MSELoss()(yPredIC, yIC)
            
            if XPhysics is None:
                lossPhysics = 0.0
            else:
                lossPhysics = self.loss_Physics(XPhysics, loss_physics_function)
            
            # Store individual losses.
            if self.first_closure_call:
                self.lossIC = lossIC.item() if isinstance(lossIC, torch.Tensor) else lossIC
                self.lossBC = lossBC.item() if isinstance(lossBC, torch.Tensor) else lossBC
                self.lossPhysics = lossPhysics.item() if isinstance(lossPhysics, torch.Tensor) else lossPhysics
                self.first_closure_call = False 
            
            # Combine losses.
            loss = 1000*lossBC + 1000*lossIC + lossPhysics
            
            # Backpropagate the loss.
            loss.backward()
            # loss.backward(retain_graph=False)

            return loss
        
        # Input-data transformation---------------------------------------------
        def to_tensor(x):
            """Convert a scalar, NumPy array, or list to a column ``float64`` tensor."""
            if x is None:
                return None
            if isinstance(x, (float, int, np.generic)):  # Scalar or NumPy scalar.
                return torch.tensor([[x]], dtype=torch.float64)
            if isinstance(x, np.ndarray):  # NumPy array.
                return torch.as_tensor(x, dtype=torch.float64).view(-1, 1)  # Ensure shape (N, 1).
            if isinstance(x, list):  # Python list.
                return torch.tensor(x, dtype=torch.float64).view(-1, 1)  # Ensure shape (N, 1).
            return x  # If it is already a tensor, leave it unchanged.
        
        XBC = to_tensor(XBC)
        yBC = to_tensor(yBC)
        XIC = to_tensor(XIC)
        yIC = to_tensor(yIC)
        XPhysics = to_tensor(XPhysics)
        
        #------------------------------------------------------------------------
        
        threshold = 1e-2 # An improvement is counted only if it exceeds this fraction.
        
        # Add ReduceLROnPlateau to the optimizer.
        if updateLR:
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=int(patience/5), threshold=threshold)
        
        for epoch in range(1,self.max_epochs+1):
                    
            if epoch > patience:
                loss_ref = self.history['Loss'].values[-patience]
                losses = self.history['Loss'].values[-patience+1:]
                improvement_max = np.max((loss_ref-losses)/loss_ref)
                if improvement_max < threshold:
                    print(f"Early stopping at epoch {epoch-1} due to insufficient improvement over the last {patience} epochs.")
                    break
           
            self.first_closure_call = True 
           
           
    
            # Start optimization.
            start_time = time.time()
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                loss = self.optimizer.step(closure)  # Perform the optimizer step.
            final_time = time.time()
            total_time = sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e6  # Convert to seconds.
            flops = sum(e.flops for e in prof.key_averages())
            self.loss = loss.item()
            
            
            self.history = pd.concat([self.history, pd.DataFrame([{
                                'Epoch': self.current_epoch, 
                                'Loss': self.loss, 
                                'LossIC': self.lossIC, 
                                'LossBC': self.lossBC, 
                                'LossPhysics': self.lossPhysics, 
                                'Time_optimizer': final_time - start_time, 
                                'Time_cpu': total_time, 
                                'Flops': flops,
                                'LR': self.optimizer.param_groups[0]['lr']
                            }])], ignore_index=True) if not self.history.empty else pd.DataFrame([{
                                'Epoch': self.current_epoch, 
                                'Loss': self.loss, 
                                'LossIC': self.lossIC, 
                                'LossBC': self.lossBC, 
                                'LossPhysics': self.lossPhysics, 
                                'Time_optimizer': final_time - start_time, 
                                'Time_cpu': total_time, 
                                'Flops': flops,
                                'LR': self.optimizer.param_groups[0]['lr']
                            }])
            
            if updateLR:
                prev_lr = scheduler.get_last_lr()
                scheduler.step(loss)
                new_lr = scheduler.get_last_lr()
                if new_lr != prev_lr:
                    print(f"Learning rate changed: {prev_lr} -> {new_lr}")
            
            print(f"Epoch {self.current_epoch}, Loss: {self.loss}, LossBC: {self.lossBC}, LossIC: {self.lossIC}, LossPhysics: {self.lossPhysics}, Time: {final_time-start_time}, FLOPS: {flops}", flush=True)
            self.current_epoch += 1
