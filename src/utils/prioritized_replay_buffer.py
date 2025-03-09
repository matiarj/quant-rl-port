import numpy as np
from typing import Tuple, List, Dict, Any
import random

class SumTree:
    """
    Binary sum tree data structure for efficient sampling based on priorities.
    Used by PrioritizedReplayBuffer for O(log n) sampling and updates.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the SumTree with a given capacity.
        
        Args:
            capacity: Maximum number of leaf nodes (replay buffer entries)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Array to store the tree
        self.data = [None] * capacity  # Array to store the data
        self.data_pointer = 0
        self.size = 0
        
    def add(self, priority: float, data: Tuple[Any, ...]) -> None:
        """
        Add a new data point with priority to the tree.
        
        Args:
            priority: Priority value for the data
            data: Data to store (state, action, reward, next_state, done)
        """
        # Find the leaf index
        tree_index = self.data_pointer + self.capacity - 1
        
        # Store the data
        self.data[self.data_pointer] = data
        
        # Update the tree with new priority
        self.update(tree_index, priority)
        
        # Move the pointer to the next leaf
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Update size
        if self.size < self.capacity:
            self.size += 1
            
    def update(self, tree_index: int, priority: float) -> None:
        """
        Update the priority of a leaf node and propagate changes up the tree.
        
        Args:
            tree_index: The index of the leaf node
            priority: The new priority value
        """
        # Change = new priority - old priority
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate the change through the tree
        while tree_index != 0:
            # Move to parent
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
            
    def get_leaf(self, value: float) -> Tuple[int, float, Tuple[Any, ...]]:
        """
        Get a leaf node based on a value.
        
        Args:
            value: Value to search for (between 0 and total priority)
            
        Returns:
            tree_index: Index of the selected leaf
            priority: Priority of the selected leaf
            data: Data associated with the selected leaf
        """
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach a leaf node
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # Otherwise, go to the left or right child based on value
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index
                
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
        
    @property
    def total_priority(self) -> float:
        """Get the total priority of the tree (root node)."""
        return self.tree[0]
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
        

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer implementation.
    Uses a sum tree data structure for efficient sampling based on TD-error priorities.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            epsilon: Small value to ensure non-zero priority even for zero TD-error
        """
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Sum tree for efficient sampling
        self.tree = np.zeros(2 * capacity - 1)
        
        # Data storage
        self.data = [None] * capacity
        self.data_pointer = 0
        self.size = 0
        
        # Maximum priority seen so far (for new transitions)
        self.max_priority = 1.0
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Create tuple of experience
        experience = (state, action, reward, next_state, done)
        
        # Find index in the tree
        index = self.data_pointer + self.capacity - 1
        
        # Store experience
        self.data[self.data_pointer] = experience
        
        # Update tree with max priority for new experience
        self.update_priority(index, self.max_priority)
        
        # Update data pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Update buffer size
        if self.size < self.capacity:
            self.size += 1
            
    def update_priority(self, index: int, td_error: float) -> None:
        """
        Update the priority of an experience based on TD-error.
        
        Args:
            index: Index of the experience in the buffer
            td_error: TD-error of the experience
        """
        # Add epsilon to ensure non-zero priority
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Update tree
        self.tree[index] = priority
        
        # Propagate the change up the tree
        parent = (index - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2]
            parent = (parent - 1) // 2
            
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no IS, 1 = full IS)
            
        Returns:
            Batch of experiences with importance sampling weights and indices
        """
        if self.size < batch_size:
            return None
            
        # Initialize arrays for batch
        states = np.zeros((batch_size, *self.data[0][0].shape), dtype=np.float32)
        actions = np.zeros((batch_size, *self.data[0][1].shape), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, *self.data[0][3].shape), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        # Importance sampling weights
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Indices to update later
        indices = np.zeros(batch_size, dtype=np.int32)
        
        # Calculate segment size
        segment_size = self.tree[0] / batch_size
        
        # Calculate importance sampling weights
        min_prob = np.min(self.tree[self.capacity-1:self.capacity+self.size-1]) / self.tree[0]
        max_weight = (min_prob * self.size) ** (-beta)
        
        # Sample from each segment
        for i in range(batch_size):
            # Get a random value within the segment
            a, b = segment_size * i, segment_size * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get experience from the tree
            index, priority, data = self._get_leaf(value)
            
            # Calculate the probability of this experience
            prob = priority / self.tree[0]
            
            # Calculate importance sampling weight
            weights[i] = (prob * self.size) ** (-beta) / max_weight
            indices[i] = index
            
            # Store experience in batch
            states[i] = data[0]
            actions[i] = data[1]
            rewards[i] = data[2]
            next_states[i] = data[3]
            dones[i] = data[4]
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def _get_leaf(self, value: float) -> Tuple[int, float, Tuple]:
        """
        Get a leaf node from the tree based on a value.
        
        Args:
            value: Value to search for (between 0 and total priority)
            
        Returns:
            index: Index of the leaf
            priority: Priority of the leaf
            data: Experience stored at the leaf
        """
        parent = 0
        
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            
            # If we reach beyond the tree
            if left_child >= len(self.tree):
                leaf_index = parent
                break
                
            # Go left or right based on comparison with left child
            if value <= self.tree[left_child]:
                parent = left_child
            else:
                value -= self.tree[left_child]
                parent = right_child
        
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for a batch."""
        return self.size >= batch_size