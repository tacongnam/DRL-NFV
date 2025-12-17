"""
DC Transition Buffer for Online GenAI Training
"""

import numpy as np
from collections import deque

class DCTransitionBuffer:
    """
    Buffer for storing DC transitions for GenAI training
    
    Stores: (dc_current_state, dc_next_state, value)
    """
    
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, dc_current, dc_next, value):
        """
        Add a DC transition
        
        Args:
            dc_current: DC state before action (18D vector)
            dc_next: DC state after action (18D vector)
            value: Calculated value for this DC
        """
        self.buffer.append({
            'current': np.array(dc_current, dtype=np.float32),
            'next': np.array(dc_next, dtype=np.float32),
            'value': float(value)
        })
    
    def sample(self, indices):
        """
        Sample batch by indices
        
        Args:
            indices: Array of indices to sample
            
        Returns:
            dict with 'current', 'next', 'value' as numpy arrays
        """
        batch = [self.buffer[i] for i in indices]
        
        return {
            'current': np.array([b['current'] for b in batch], dtype=np.float32),
            'next': np.array([b['next'] for b in batch], dtype=np.float32),
            'value': np.array([b['value'] for b in batch], dtype=np.float32)
        }
    
    def sample_random(self, batch_size):
        """
        Sample random batch
        
        Args:
            batch_size: Number of samples
            
        Returns:
            dict with 'current', 'next', 'value' as numpy arrays
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return self.sample(indices)
    
    def get_all(self):
        """Get all data (for full training)"""
        if len(self.buffer) == 0:
            return {
                'current': np.array([], dtype=np.float32),
                'next': np.array([], dtype=np.float32),
                'value': np.array([], dtype=np.float32)
            }
        
        return {
            'current': np.array([b['current'] for b in self.buffer], dtype=np.float32),
            'next': np.array([b['next'] for b in self.buffer], dtype=np.float32),
            'value': np.array([b['value'] for b in self.buffer], dtype=np.float32)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def get_stats(self):
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'mean_value': 0.0,
                'std_value': 0.0
            }
        
        values = np.array([b['value'] for b in self.buffer])
        
        return {
            'size': len(self.buffer),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values)
        }