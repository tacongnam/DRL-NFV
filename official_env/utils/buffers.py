import numpy as np
import random
from collections import deque

class ReplayBuffer:
    # (Giữ nguyên cho DQN)
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def size(self):
        return len(self.buffer)

class VAEDataset:
    """
    [Paper Fig. 2]: Dataset collects (DC State, DC Next State, Value)
    """
    def __init__(self):
        self.current_states = []
        self.next_states = []
        self.values = [] 

    def add(self, curr, next_s, val):
        self.current_states.append(curr)
        self.next_states.append(next_s)
        self.values.append(val)
        
    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.current_states), batch_size)
        
        c_batch = np.array(self.current_states)[indices]
        n_batch = np.array(self.next_states)[indices]
        v_batch = np.array(self.values)[indices]
        
        return c_batch, n_batch, v_batch