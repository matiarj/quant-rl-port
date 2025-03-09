import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import os
import time
import copy

from src.agents.base_agent import BaseAgent, BaseNetwork
from src.utils.replay_buffer import ReplayBuffer

class DDQN(BaseAgent):
    """Double Deep Q-Network (DDQN) agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize DDQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
        
        # Initialize hyperparameters
        self.lr = config['network']['learning_rate']
        self.gamma = config['rl']['gamma']
        self.batch_size = config['rl']['batch_size']
        self.target_update = config['rl']['target_update']
        
        # Epsilon-greedy exploration
        self.eps_start = config['rl']['eps_start']
        self.eps_end = config['rl']['eps_end']
        self.eps_decay = config['rl']['eps_decay']
        self.epsilon = self.eps_start
        
        # Network architecture parameters
        self.hidden_dims = config['network']['hidden_sizes']
        self.dropout_rate = config['network'].get('dropout', 0.0)
        self.use_batch_norm = config['network'].get('use_batch_norm', False)
        self.training = True  # Track training mode
        
        # Build networks
        self.online_network = BaseNetwork(
            state_dim, 
            action_dim, 
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        self.target_network = BaseNetwork(
            state_dim, 
            action_dim, 
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # Initialize target network with same weights as online network
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(config['rl']['memory_size'])
        
        # Training steps counter for target network updates
        self.steps_done = 0
        
        self.logger.info(f"Initialized DDQN agent with hidden dims {self.hidden_dims}, lr={self.lr}, gamma={self.gamma}")
        self.logger.info(f"Target network update frequency: {self.target_update}")
    
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            action: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Initialize action with zeros (no action)
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        if explore and np.random.rand() < self.epsilon:
            # Random action
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Use online network to select best action (in eval mode for batch norm)
            self.online_network.eval()
            with torch.no_grad():
                q_values = self.online_network(state_tensor)
                # Scale Q-values to action range [-1, 1]
                scaled_q_values = torch.tanh(q_values)
                action = scaled_q_values.cpu().squeeze().numpy()
            # Switch back to train mode if needed
            if self.training:
                self.online_network.train()
        
        return action
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Dictionary of loss values and metrics
        """
        # Check if enough samples in replay buffer
        if not self.memory.is_ready(self.batch_size):
            return {"loss": 0.0}
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s,a) from online network
        q_values = self.online_network(states)
        
        # Compute the action values by dot product of q_values and actions
        state_action_values = torch.sum(q_values * actions, dim=1)
        
        # Double DQN: use online network to select actions and target network to evaluate them
        with torch.no_grad():
            # Select actions using online network
            next_q_values_online = self.online_network(next_states)
            best_actions = torch.argmax(next_q_values_online, dim=1, keepdim=True)
            
            # Evaluate selected actions using target network
            next_q_values_target = self.target_network(next_states)
            next_state_values = next_q_values_target.gather(1, best_actions).squeeze()
            
            # Calculate expected Q values (Bellman equation)
            expected_q_values = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(state_action_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increment steps and update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
            
        # Update epsilon for exploration
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        return {
            "loss": loss.item(), 
            "epsilon": self.epsilon,
            "steps": self.steps_done
        }
    
    def update_target_network(self) -> None:
        """Update target network with current online network weights."""
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.logger.debug(f"Target network updated at step {self.steps_done}")
    
    def save_model(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)
        timestamp = int(time.time())
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'hidden_dims': self.hidden_dims,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, os.path.join(path, f'ddqn_model_{timestamp}.pth'))
        self.logger.info(f"Model saved to {path}/ddqn_model_{timestamp}.pth")
    
    def load_model(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            self.logger.error(f"Model path {path} does not exist")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify state and action dimensions match
        if self.state_dim != checkpoint.get('state_dim') or self.action_dim != checkpoint.get('action_dim'):
            self.logger.error(f"Loaded model has different dimensions than the current environment")
            return
        
        # Load network parameters
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        self.logger.info(f"Model loaded from {path}")