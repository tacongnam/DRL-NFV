import numpy as np
import random
from collections import deque

class VAEBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, next_state, value):
        self.buffer.append((state, next_state, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, ns, v = zip(*batch)
        return np.array(s), np.array(ns), np.array(v)
    
    def __len__(self):
        return len(self.buffer)

class DQNBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, dc_s, loc_s, glob_s, action, reward, n_dc_s, n_loc_s, n_glob_s, done):
        self.buffer.append((dc_s, loc_s, glob_s, action, reward, n_dc_s, n_loc_s, n_glob_s, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        dc_s, loc_s, glob_s, a, r, n_dc_s, n_loc_s, n_glob_s, d = zip(*batch)
        return (np.array(dc_s), np.array(loc_s), np.array(glob_s), 
                np.array(a), np.array(r), 
                np.array(n_dc_s), np.array(n_loc_s), np.array(n_glob_s), np.array(d))
    
    def __len__(self):
        return len(self.buffer)