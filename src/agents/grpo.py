import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import os
import time
import logging

from src.agents.base_agent import BaseAgent

class GRPONetwork(nn.Module):
    """Network for GRPO (Gradient-based Robust Policy Optimization)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], dropout: float = 0.0):
        """
        Initialize GRPO network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.Sequential()
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.add_module(f'fc{i}', nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.add_module(f'relu{i}', nn.ReLU())
            if dropout > 0:
                self.feature_layers.add_module(f'dropout{i}', nn.Dropout(dropout))
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
        
        # Adversarial critic (for robustness) - estimates worst-case value
        self.adv_critic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Tuple of (action_values, state_value, adversarial_value)
        """
        features = self.feature_layers(x)
        action_values = self.actor_head(features)
        state_value = self.critic_head(features)
        adv_value = self.adv_critic_head(features)
        
        return action_values, state_value, adv_value


class GRPO(BaseAgent):
    """Gradient-based Robust Policy Optimization (GRPO) agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize GRPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
        
        # Initialize hyperparameters
        self.lr = config['network']['learning_rate']
        self.gamma = config['rl']['gamma']
        self.eps_clip = config.get('rl', {}).get('eps_clip', 0.2)
        self.entropy_coef = config.get('rl', {}).get('entropy_coef', 0.01)
        self.value_coef = config.get('rl', {}).get('value_coef', 0.5)
        self.adv_coef = config.get('rl', {}).get('adv_coef', 0.1)  # Weight for adversarial loss
        self.uncertainty = config.get('rl', {}).get('uncertainty', 0.05)  # Uncertainty parameter
        
        # Build network
        self.hidden_dims = config['network']['hidden_sizes']
        self.dropout = config['network'].get('dropout', 0.0)
        
        self.network = GRPONetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer with lower learning rate for stability
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Initialize learning rate scheduler if enabled
        self.use_lr_scheduler = config['network'].get('lr_scheduler', False)
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=10,
                verbose=True
            )
        
        # Memory buffer for GRPO (much smaller than PPO)
        # We only need to store the most recent transition
        self.recent_state = None
        self.recent_action = None
        self.recent_reward = None
        self.recent_next_state = None
        self.recent_done = None
        
        # Mini-batch storage (very small, just for efficiency)
        self.mini_batch_size = config.get('rl', {}).get('mini_batch_size', 8)
        self.states_batch = []
        self.actions_batch = []
        self.rewards_batch = []
        self.next_states_batch = []
        self.dones_batch = []
        
        # Track training mode
        self.training = True
        
        # Create dummy memory attribute for compatibility with the ensemble code
        class StubMemory:
            def add(self, *args, **kwargs):
                pass
                
            def is_ready(self, batch_size):
                return False
        
        self.memory = StubMemory()
        
        # Running statistics for value and advantage normalization
        self.value_mean = 0
        self.value_std = 1
        self.adv_mean = 0
        self.adv_std = 1
        self.value_count = 0
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized GRPO agent with hidden dims {self.hidden_dims}, "
            f"lr={self.lr}, gamma={self.gamma}, eps_clip={self.eps_clip}, "
            f"entropy_coef={self.entropy_coef}, value_coef={self.value_coef}, "
            f"adv_coef={self.adv_coef}, uncertainty={self.uncertainty}"
        )
    
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
        self.network.eval()
        with torch.no_grad():
            action_values, _, _ = self.network(state_tensor)
        if hasattr(self, 'training') and self.training:
            self.network.train()
            
        # Convert to numpy array
        action = action_values.cpu().squeeze().numpy()
        
        # Add exploration noise if required
        if explore:
            # Adaptive noise based on uncertainty (lower as training progresses)
            current_uncertainty = self.uncertainty * (1.0 / (1.0 + 0.01 * self.train_steps))
            noise = np.random.normal(0, current_uncertainty, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience for GRPO update.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store the most recent transition
        self.recent_state = state
        self.recent_action = action
        self.recent_reward = reward
        self.recent_next_state = next_state
        self.recent_done = done
        
        # Add to mini-batch for efficiency
        self.states_batch.append(state)
        self.actions_batch.append(action)
        self.rewards_batch.append(reward)
        self.next_states_batch.append(next_state)
        self.dones_batch.append(done)
        
        # If mini-batch is full, update immediately (online learning)
        if len(self.states_batch) >= self.mini_batch_size:
            self.train_step()
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single GRPO update.
        
        Returns:
            Dictionary of loss values and metrics
        """
        # Check if we have any experiences
        if not self.states_batch:
            return {"loss": 0.0}
        
        # Convert stored experiences to tensors
        states = torch.FloatTensor(np.array(self.states_batch)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions_batch)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards_batch)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states_batch)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones_batch)).to(self.device)
        
        # Clear mini-batch
        self.states_batch = []
        self.actions_batch = []
        self.rewards_batch = []
        self.next_states_batch = []
        self.dones_batch = []
        
        # Get current policy's action values and state values
        action_values, state_values, adv_values = self.network(states)
        state_values = state_values.squeeze()
        adv_values = adv_values.squeeze()
        
        # Compute next state values for TD targets
        with torch.no_grad():
            _, next_state_values, _ = self.network(next_states)
            next_state_values = next_state_values.squeeze()
            next_state_values = next_state_values * (1 - dones)
        
        # Compute TD targets and advantages
        td_targets = rewards + self.gamma * next_state_values
        advantages = td_targets - state_values
        
        # Compute robust advantages (worst-case estimates)
        robust_advantages = advantages - self.uncertainty * torch.abs(advantages)
        
        # Compute actor loss using robust advantages
        weighted_action_values = torch.sum(action_values * actions, dim=1)
        actor_loss = -(weighted_action_values * robust_advantages.detach()).mean()
        
        # Compute critic loss for value function (MSE)
        critic_loss = F.mse_loss(state_values, td_targets.detach())
        
        # Compute adversarial critic loss (to learn worst-case values)
        # Target is the regular value minus uncertainty
        adv_targets = td_targets - self.uncertainty * torch.abs(td_targets - state_values.detach())
        adv_critic_loss = F.mse_loss(adv_values, adv_targets.detach())
        
        # Add entropy bonus to encourage exploration
        action_probs = (action_values + 1) / 2  # Convert from [-1,1] to [0,1]
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
        
        # Compute total loss with coefficients
        total_loss = (
            actor_loss + 
            self.value_coef * critic_loss + 
            self.adv_coef * adv_critic_loss - 
            self.entropy_coef * entropy
        )
        
        # Update running statistics for normalization
        with torch.no_grad():
            self.value_count += len(state_values)
            delta_mean = (td_targets.mean().item() - self.value_mean) / self.value_count
            self.value_mean += delta_mean
            delta_std = (td_targets.std().item() - self.value_std) / self.value_count
            self.value_std = max(1e-6, self.value_std + delta_std)
            
            adv_mean_new = advantages.mean().item()
            adv_std_new = max(1e-6, advantages.std().item())
            self.adv_mean = 0.9 * self.adv_mean + 0.1 * adv_mean_new
            self.adv_std = 0.9 * self.adv_std + 0.1 * adv_std_new
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        
        self.optimizer.step()
        self.train_steps += 1
        
        # Update learning rate if scheduler is enabled
        if self.use_lr_scheduler and self.train_steps % 100 == 0:
            self.scheduler.step(rewards.mean())
        
        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "adv_critic_loss": adv_critic_loss.item(),
            "entropy": entropy.item(),
            "advantage_mean": advantages.mean().item(),
            "robust_advantage_mean": robust_advantages.mean().item()
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
            'action_dim': self.action_dim,
            'value_mean': self.value_mean,
            'value_std': self.value_std,
            'adv_mean': self.adv_mean,
            'adv_std': self.adv_std,
            'train_steps': self.train_steps
        }, os.path.join(path, f'grpo_model_{timestamp}.pth'))
        self.logger.info(f"Model saved to {path}/grpo_model_{timestamp}.pth")
    
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
        
        # Load running statistics
        self.value_mean = checkpoint.get('value_mean', 0)
        self.value_std = checkpoint.get('value_std', 1)
        self.adv_mean = checkpoint.get('adv_mean', 0)
        self.adv_std = checkpoint.get('adv_std', 1) 
        self.train_steps = checkpoint.get('train_steps', 0)
        
        self.logger.info(f"Model loaded from {path}")
    
    def update_target_network(self) -> None:
        """
        For GRPO, there is no explicit target network update.
        This method is here for compatibility with other agents.
        """
        pass