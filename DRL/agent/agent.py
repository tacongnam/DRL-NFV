import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import config
from agent.model import build_q_network

class Agent:
    def __init__(self):
        # Khởi tạo Main Model và Target Model
        self.model = build_q_network()
        self.target_model = build_q_network()
        self.update_target_model()

    def update_target_model(self):
        """Sao chép trọng số từ model chính sang target model"""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon, valid_actions_mask=None):
        """
        Chọn hành động dựa trên Epsilon-Greedy và Action Masking.
        
        Args:
            state: Tuple 3 phần tử (state_1, state_2, state_3)
            epsilon: Xác suất exploration
            valid_actions_mask: Mảng boolean (True = hành động hợp lệ)
        """
        if np.random.rand() <= epsilon:
            # Random exploration - but only among VALID actions
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(config.ACTION_SPACE_SIZE)
        
        # Exploitation - get Q-values from model
        # Reshape inputs to (1, feature_size) for batch prediction
        state_1 = tf.convert_to_tensor(state[0].reshape(1, -1))
        state_2 = tf.convert_to_tensor(state[1].reshape(1, -1))
        state_3 = tf.convert_to_tensor(state[2].reshape(1, -1))
        
        q_values = self.model([state_1, state_2, state_3], training=False)
        q_values = q_values.numpy()[0]
        
        # Apply action masking
        if valid_actions_mask is not None:
            # Set invalid actions to very negative value to avoid selection
            masked_q_values = np.where(valid_actions_mask, q_values, -1e9)
            return int(np.argmax(masked_q_values))
        
        return int(np.argmax(q_values))

    def train(self, replay_memory):
        mem_size = len(replay_memory)
        if mem_size < config.BATCH_SIZE:
            return 0.0  # Not enough samples to train

        batch_indices = np.random.choice(mem_size, config.BATCH_SIZE, replace=False)
        minibatch = [replay_memory[i] for i in batch_indices]
        
        # Prepare batches
        # Extract components from the batch tuples: (state, action, reward, next_state, done)
        states_1 = np.array([i[0][0] for i in minibatch])
        states_2 = np.array([i[0][1] for i in minibatch])
        states_3 = np.array([i[0][2] for i in minibatch])
        
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        
        next_states_1 = np.array([i[3][0] for i in minibatch])
        next_states_2 = np.array([i[3][1] for i in minibatch])
        next_states_3 = np.array([i[3][2] for i in minibatch])
        
        dones = np.array([i[4] for i in minibatch])

        # Predict Q-values for next states using Target Model
        target_qs = self.target_model.predict([next_states_1, next_states_2, next_states_3], verbose=0)
        
        # Predict Q-values for current states using Main Model
        targets = self.model.predict([states_1, states_2, states_3], verbose=0)

        # Bellman Equation Update
        for i in range(config.BATCH_SIZE):
            target_val = rewards[i]
            if not dones[i]:
                target_val += config.GAMMA * np.max(target_qs[i])
            targets[i][actions[i]] = target_val

        # Train the model
        history = self.model.fit(
            [states_1, states_2, states_3], targets, 
            batch_size=config.BATCH_SIZE, verbose=0, epochs=1
        )
        return history.history['loss'][0]