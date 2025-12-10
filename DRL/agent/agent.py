# agent/agent.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import config
from agent.model import build_q_network

class Agent:
    """DQN Agent cho SFC Provisioning"""
    
    def __init__(self):
        self.model = build_q_network()
        self.target_model = build_q_network()
        self.update_target_model()
    
    def update_target_model(self):
        """Copy weights từ model sang target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_action(self, state, epsilon, valid_actions_mask=None):
        """
        Chọn action theo epsilon-greedy policy
        
        Args:
            state: Tuple (s1, s2, s3)
            epsilon: Exploration rate
            valid_actions_mask: Boolean array indicating valid actions
            
        Returns:
            int: Action ID
        """
        # Exploration
        if np.random.rand() <= epsilon:
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                return np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
            return np.random.randint(config.ACTION_SPACE_SIZE)
        
        # Exploitation
        # Prepare inputs
        s1 = state[0].reshape(1, -1)
        s2 = state[1].reshape(1, -1)
        s3 = state[2].reshape(1, -1)
        
        # Predict Q-values
        q_values = self.model.predict([s1, s2, s3], verbose=0)[0]
        
        # Apply mask
        if valid_actions_mask is not None:
            q_values = np.where(valid_actions_mask, q_values, -np.inf)
        
        return np.argmax(q_values)
    
    def train(self, replay_memory):
        """
        Train model trên batch từ replay memory
        
        Args:
            replay_memory: Deque chứa transitions
            
        Returns:
            float: Loss value
        """
        if len(replay_memory) < config.BATCH_SIZE:
            return 0.0
        
        # Sample batch
        batch_indices = np.random.choice(len(replay_memory), 
                                        config.BATCH_SIZE, 
                                        replace=False)
        
        # Get shapes from first sample
        sample_state = replay_memory[0][0]
        s1_shape = sample_state[0].shape
        s2_shape = sample_state[1].shape
        s3_shape = sample_state[2].shape
        
        # Initialize batches
        state1_batch = np.zeros((config.BATCH_SIZE, *s1_shape), dtype=np.float32)
        state2_batch = np.zeros((config.BATCH_SIZE, *s2_shape), dtype=np.float32)
        state3_batch = np.zeros((config.BATCH_SIZE, *s3_shape), dtype=np.float32)
        
        next_state1_batch = np.zeros((config.BATCH_SIZE, *s1_shape), dtype=np.float32)
        next_state2_batch = np.zeros((config.BATCH_SIZE, *s2_shape), dtype=np.float32)
        next_state3_batch = np.zeros((config.BATCH_SIZE, *s3_shape), dtype=np.float32)
        
        action_batch = np.zeros(config.BATCH_SIZE, dtype=np.int32)
        reward_batch = np.zeros(config.BATCH_SIZE, dtype=np.float32)
        done_batch = np.zeros(config.BATCH_SIZE, dtype=bool)
        
        # Fill batches
        for i, idx in enumerate(batch_indices):
            # transition: (state, action, reward, next_state, done)
            state, action, reward, next_state, done = replay_memory[idx]
            
            state1_batch[i] = state[0]
            state2_batch[i] = state[1]
            state3_batch[i] = state[2]
            
            action_batch[i] = action
            reward_batch[i] = reward
            
            next_state1_batch[i] = next_state[0]
            next_state2_batch[i] = next_state[1]
            next_state3_batch[i] = next_state[2]
            
            done_batch[i] = done
        
        # Predict next Q-values using target network
        next_q_values = self.target_model.predict(
            [next_state1_batch, next_state2_batch, next_state3_batch],
            verbose=0
        )
        max_next_q = np.amax(next_q_values, axis=1)
        
        # Calculate targets
        targets = reward_batch + (1 - done_batch) * config.GAMMA * max_next_q
        
        # Get current Q-values
        current_q_values = self.model.predict(
            [state1_batch, state2_batch, state3_batch],
            verbose=0
        )
        
        # Update Q-values for taken actions
        indices = np.arange(config.BATCH_SIZE)
        current_q_values[indices, action_batch] = targets
        
        # Train
        history = self.model.fit(
            [state1_batch, state2_batch, state3_batch],
            current_q_values,
            batch_size=config.BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        
        return history.history['loss'][0]