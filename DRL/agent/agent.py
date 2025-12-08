# agent/agent.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import config
from agent.model import build_q_network

class Agent:
    def __init__(self):
        self.model = build_q_network()
        self.target_model = build_q_network()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon, valid_actions_mask=None):
        # state là tuple (s1, s2, s3)
        # Cần reshape để đưa vào model: [1, dim1], [1, dim2], [1, dim3]
        
        if np.random.rand() <= epsilon:
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                return np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
            return np.random.randint(config.ACTION_SPACE_SIZE)

        # Prepare inputs
        s1 = state[0].reshape(1, -1)
        s2 = state[1].reshape(1, -1)
        s3 = state[2].reshape(1, -1)
        
        q_values = self.model.predict([s1, s2, s3], verbose=0)[0]

        # Masking
        if valid_actions_mask is not None:
            q_values = np.where(valid_actions_mask, q_values, -np.inf)
            
        return np.argmax(q_values)

    def train(self, replay_memory):
        if len(replay_memory) < config.BATCH_SIZE:
            return 0.0

        batch_indices = np.random.choice(len(replay_memory), config.BATCH_SIZE, replace=False)
        
        # Init batches
        # Lấy shape từ sample đầu tiên
        s1_shape = replay_memory[0][0][0].shape
        s2_shape = replay_memory[0][0][1].shape
        s3_shape = replay_memory[0][0][2].shape
        
        state1_batch = np.zeros((config.BATCH_SIZE, *s1_shape))
        state2_batch = np.zeros((config.BATCH_SIZE, *s2_shape))
        state3_batch = np.zeros((config.BATCH_SIZE, *s3_shape))
        
        next_state1_batch = np.zeros((config.BATCH_SIZE, *s1_shape))
        next_state2_batch = np.zeros((config.BATCH_SIZE, *s2_shape))
        next_state3_batch = np.zeros((config.BATCH_SIZE, *s3_shape))
        
        action_batch = np.zeros(config.BATCH_SIZE, dtype=np.int32)
        reward_batch = np.zeros(config.BATCH_SIZE, dtype=np.float32)
        done_batch = np.zeros(config.BATCH_SIZE, dtype=bool)

        for i, idx in enumerate(batch_indices):
            # transition: (state, action, reward, next_state, done)
            # state is (s1, s2, s3)
            transition = replay_memory[idx]
            
            state1_batch[i] = transition[0][0]
            state2_batch[i] = transition[0][1]
            state3_batch[i] = transition[0][2]
            
            action_batch[i] = transition[1]
            reward_batch[i] = transition[2]
            
            next_state1_batch[i] = transition[3][0]
            next_state2_batch[i] = transition[3][1]
            next_state3_batch[i] = transition[3][2]
            
            done_batch[i] = transition[4]

        # Predict Next Q using Target
        next_q = self.target_model.predict(
            [next_state1_batch, next_state2_batch, next_state3_batch], 
            verbose=0
        )
        max_next_q = np.amax(next_q, axis=1)
        
        targets = reward_batch + (1 - done_batch) * config.GAMMA * max_next_q
        
        # Predict Current Q to update
        target_f = self.model.predict(
            [state1_batch, state2_batch, state3_batch], 
            verbose=0
        )
        
        # Update target values
        indices = np.arange(config.BATCH_SIZE)
        target_f[indices, action_batch] = targets
        
        # Train
        history = self.model.fit(
            [state1_batch, state2_batch, state3_batch],
            target_f,
            batch_size=config.BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        return history.history['loss'][0]