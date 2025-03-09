import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import os
import time

from src.agents.base_agent import BaseAgent, BaseNetwork
from src.utils.replay_buffer import ReplayBuffer

class DQN(BaseAgent):
    """Deep Q-Network (DQN) agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize DQN agent.
        
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
        self.q_network = BaseNetwork(
            state_dim, 
            action_dim, 
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # Target network for stable training
        self.target_network = BaseNetwork(
            state_dim, 
            action_dim, 
            self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # Copy parameters from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Target network update frequency
        self.target_update = config['rl']['target_update']
        self.update_counter = 0
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Learning rate scheduler if enabled
        self.scheduler = None
        if config['network'].get('lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=100,
                cooldown=50, 
                min_lr=1e-6,
                verbose=True
            )
        
        # Initialize replay buffer
        memory_size = config['rl']['memory_size']
        self.prioritized_replay = config['rl'].get('prioritized_replay', False)
        
        if self.prioritized_replay:
            from src.utils.prioritized_replay_buffer import PrioritizedReplayBuffer
            self.memory = PrioritizedReplayBuffer(memory_size, alpha=0.6)
            self.beta = 0.4  # Start value for importance sampling
            self.beta_increment = 0.001
        else:
            self.memory = ReplayBuffer(memory_size)
            
        self.logger.info(f"Initialized DQN agent with:")
        self.logger.info(f"  - Hidden dims: {self.hidden_dims}")
        self.logger.info(f"  - Learning rate: {self.lr}")
        self.logger.info(f"  - Gamma: {self.gamma}")
        self.logger.info(f"  - Batch size: {self.batch_size}")
        self.logger.info(f"  - Dropout: {self.dropout_rate}")
        self.logger.info(f"  - Batch norm: {self.use_batch_norm}")
        self.logger.info(f"  - Prioritized replay: {self.prioritized_replay}")
    
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
            # Use Q-network to select best action (set to eval mode to disable BatchNorm)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # Scale Q-values to action range [-1, 1]
                scaled_q_values = torch.tanh(q_values)
                action = scaled_q_values.cpu().squeeze().numpy()
            # Switch back to train mode if we're training
            if self.training:
                self.q_network.train()
        
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
        if self.prioritized_replay:
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Get sample with importance sampling weights
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(
                self.batch_size, beta=self.beta
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights, indices = None, None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s,a)
        q_values = self.q_network(states)
        
        # We need to reshape q_values to match actions shape for proper element-wise multiplication
        # q_values shape: [batch_size, action_dim]
        # actions shape: [batch_size, action_dim]
        
        # Compute the action values by dot product of q_values and actions
        # This gives us the weighted Q-value for the actual actions taken
        state_action_values = torch.sum(q_values * actions, dim=1)
        
        # Compute V(s') for all next states using target network
        with torch.no_grad():
            # Use target network for stable learning
            next_q_values = self.target_network(next_states)
            best_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
            next_state_values = next_q_values.gather(1, best_actions).squeeze()
            
            # Calculate expected Q values (Bellman equation)
            expected_q_values = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Compute loss with importance sampling if using prioritized replay
        if self.prioritized_replay:
            # Element-wise loss for prioritized replay
            td_errors = torch.abs(state_action_values - expected_q_values).detach().cpu().numpy()
            
            # Update priorities in the replay buffer
            for i in range(self.batch_size):
                self.memory.update_priority(indices[i], td_errors[i])
            
            # Use weighted MSE loss (multiply element-wise losses by weights)
            elementwise_loss = F.mse_loss(state_action_values, expected_q_values, reduction='none')
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = F.mse_loss(state_action_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update learning rate if scheduler is enabled
        if self.scheduler is not None:
            self.scheduler.step(loss)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.update_target_network()
            
        # Update epsilon for exploration
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        return {
            "loss": loss.item(), 
            "epsilon": self.epsilon,
            "lr": self.optimizer.param_groups[0]['lr']
        }
    
    def save_model(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)
        timestamp = int(time.time())
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'hidden_dims': self.hidden_dims,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, os.path.join(path, f'dqn_model_{timestamp}.pth'))
        self.logger.info(f"Model saved to {path}/dqn_model_{timestamp}.pth")
    
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
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        self.logger.info(f"Model loaded from {path}")
        
    def update_target_network(self) -> None:
        """
        Update target network by copying parameters from the
        online network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("Target network updated")