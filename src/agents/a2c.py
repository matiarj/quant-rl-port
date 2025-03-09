import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import os
import time

from src.agents.base_agent import BaseAgent
from src.utils.replay_buffer import ReplayBuffer

class ActorCritic(nn.Module):
    """Actor-Critic network for A2C algorithm."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.Sequential()
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):  # Use all but the last hidden layer
            self.feature_layers.add_module(f'fc{i}', nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.add_module(f'relu{i}', nn.ReLU())
            prev_dim = hidden_dim
        
        # Actor network (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # Tanh to bound actions between -1 and 1
        )
        
        # Critic network (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Tuple of (action_values, state_value)
        """
        features = self.feature_layers(x)
        action_values = self.actor_head(features)
        state_value = self.critic_head(features)
        
        return action_values, state_value


class A2C(BaseAgent):
    """Advantage Actor-Critic (A2C) agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize A2C agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
        
        # Initialize hyperparameters
        self.lr = config['network']['learning_rate']
        self.gamma = config['rl']['gamma']
        
        # Build network
        self.hidden_dims = config['network']['hidden_sizes']
        self.network = ActorCritic(state_dim, action_dim, self.hidden_dims).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Initialize memory (for on-policy learning)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Track training mode
        self.training = True
        
        # Create dummy memory attribute for compatibility with the ensemble code
        # A2C is on-policy and doesn't use a replay buffer, but we need this for interface compatibility
        class StubMemory:
            def add(self, *args, **kwargs):
                pass
                
            def is_ready(self, batch_size):
                return False
        
        self.memory = StubMemory()
        
        self.logger.info(f"Initialized A2C agent with hidden dims {self.hidden_dims}, lr={self.lr}, gamma={self.gamma}")
    
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action using the policy network.
        
        Args:
            state: Current state
            explore: Whether to add exploration noise
            
        Returns:
            action: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from policy network
        self.network.eval()  # Set to eval mode for batch norm
        with torch.no_grad():
            action_values, _ = self.network(state_tensor)
        if hasattr(self, 'training') and self.training:
            self.network.train()
            
        # Convert to numpy array
        action = action_values.cpu().squeeze().numpy()
        
        # Add exploration noise if required
        if explore:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step (policy update).
        
        Returns:
            Dictionary of loss values and metrics
        """
        # Check if we have any experiences
        if len(self.states) == 0:
            return {"loss": 0.0}
        
        # Convert stored experiences to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Compute action values and state values
        action_values, state_values = self.network(states)
        state_values = state_values.squeeze()
        
        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            next_state_values = next_state_values.squeeze()
            next_state_values = next_state_values * (1 - dones)
        
        # Compute advantages
        advantages = rewards + self.gamma * next_state_values - state_values
        
        # Compute critic loss (MSE)
        critic_loss = advantages.pow(2).mean()
        
        # Compute actor loss
        # We create a pseudo-loss by trying to maximize action values weighted by advantages
        # Note: The action values are already between -1 and 1 (tanh output)
        weighted_action_values = torch.sum(action_values * actions, dim=1)
        actor_loss = -(weighted_action_values * advantages.detach()).mean()
        
        # Combine losses
        total_loss = actor_loss + critic_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "advantage_mean": advantages.mean().item()
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
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_dims': self.hidden_dims,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, os.path.join(path, f'a2c_model_{timestamp}.pth'))
        self.logger.info(f"Model saved to {path}/a2c_model_{timestamp}.pth")
    
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
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {path}")
    
    def update_target_network(self) -> None:
        """
        For A2C, there is no explicit target network update.
        This method is here for compatibility with other agents.
        """
        pass