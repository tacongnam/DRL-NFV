import numpy as np
from collections import deque
import random
from config import TRAINING_CONFIG

class ReplayBuffer:
    def __init__(self, capacity=TRAINING_CONFIG['buffer_size']):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, dc_state, sfc_state, network_state, action, reward, 
             next_dc_state, next_sfc_state, next_network_state, done):
        self.buffer.append((dc_state, sfc_state, network_state, action, reward,
                          next_dc_state, next_sfc_state, next_network_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        dc_states, sfc_states, network_states = [], [], []
        actions, rewards = [], []
        next_dc_states, next_sfc_states, next_network_states = [], [], []
        dones = []
        
        for transition in batch:
            dc_s, sfc_s, net_s, a, r, next_dc_s, next_sfc_s, next_net_s, d = transition
            dc_states.append(dc_s)
            sfc_states.append(sfc_s)
            network_states.append(net_s)
            actions.append(a)
            rewards.append(r)
            next_dc_states.append(next_dc_s)
            next_sfc_states.append(next_sfc_s)
            next_network_states.append(next_net_s)
            dones.append(d)
        
        return (
            (np.array(dc_states), np.array(sfc_states), np.array(network_states)),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            (np.array(next_dc_states), np.array(next_sfc_states), np.array(next_network_states)),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class VAEDataset:
    def __init__(self, capacity=TRAINING_CONFIG['buffer_size']):
        self.capacity = capacity
        self.current_states = []
        self.next_states = []
        self.values = []
    
    def add(self, current_state, next_state, value):
        self.current_states.append(current_state)
        self.next_states.append(next_state)
        self.values.append(value)
        
        if len(self.current_states) > self.capacity:
            self.current_states.pop(0)
            self.next_states.pop(0)
            self.values.pop(0)
    
    def get_batch(self, batch_size):
        if len(self.current_states) < batch_size:
            return None
        
        indices = np.random.choice(len(self.current_states), batch_size, replace=False)
        
        current_batch = np.array([self.current_states[i] for i in indices])
        next_batch = np.array([self.next_states[i] for i in indices])
        value_batch = np.array([[self.values[i]] for i in indices])
        
        return current_batch, next_batch, value_batch
    
    def __len__(self):
        return len(self.current_states)