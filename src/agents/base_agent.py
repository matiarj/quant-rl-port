import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import List, Tuple, Dict, Any, Optional
import os
import time

class BaseAgent:
    """Base class for all RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logger()
        
        self.logger.info(f"Initialized agent with state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'logs/{self.__class__.__name__}.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            explore: Whether to explore or exploit
            
        Returns:
            action: Selected action
        """
        raise NotImplementedError("Each agent must implement an act method.")
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Dictionary of loss values and metrics
        """
        raise NotImplementedError("Each agent must implement a train_step method.")
    
    def save_model(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save the model
        """
        raise NotImplementedError("Each agent must implement a save_model method.")
    
    def load_model(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
        """
        raise NotImplementedError("Each agent must implement a load_model method.")
        

class BaseNetwork(nn.Module):
    """Base neural network architecture for RL agents."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 dropout_rate: float = 0.0, use_batch_norm: bool = False):
        """
        Initialize base network.
        
        Args:
            input_dim: Dimension of input (state)
            output_dim: Dimension of output (action values or policy)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability (0.0 means no dropout)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Optional dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)