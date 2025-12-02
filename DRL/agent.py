import numpy as np
from dqn_model import DQNModel
from utils import ReplayBuffer
from config import DRL_CONFIG

class DQNAgent:
    def __init__(self, state1_dim, state2_dim, state3_dim, action_dim):
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.state3_dim = state3_dim
        self.action_dim = action_dim
        
        self.model = DQNModel(state1_dim, state2_dim, state3_dim, action_dim)
        self.memory = ReplayBuffer(DRL_CONFIG['memory_size'])
        
        self.epsilon = DRL_CONFIG['epsilon_start']
        self.epsilon_min = DRL_CONFIG['epsilon_end']
        self.epsilon_decay = DRL_CONFIG['epsilon_decay']
        
        self.update_counter = 0
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state1 = np.expand_dims(state['state1'], axis=0)
        state2 = np.expand_dims(state['state2'], axis=0)
        state3 = np.expand_dims(state['state3'], axis=0)
        
        q_values = self.model.predict(state1, state2, state3)
        return np.argmax(q_values[0])
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(
            state['state1'], state['state2'], state['state3'],
            action, reward,
            next_state['state1'], next_state['state2'], next_state['state3'],
            done
        )
    
    def train(self):
        if len(self.memory) < DRL_CONFIG['batch_size']:
            return None
        
        s1, s2, s3, actions, rewards, ns1, ns2, ns3, dones = self.memory.sample(
            DRL_CONFIG['batch_size']
        )
        
        loss = self.model.train_on_batch(s1, s2, s3, actions, rewards, ns1, ns2, ns3, dones)
        
        self.update_counter += 1
        if self.update_counter % DRL_CONFIG['target_update_freq'] == 0:
            self.model.update_target_model()
        
        return loss
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model.load(path)