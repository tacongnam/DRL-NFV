# agent/agent.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from DRL import config
from DRL.agents.dqn_model import build_q_network
from collections import deque

class Agent:
    """DQN Agent cho SFC Provisioning"""
    
    def __init__(self):
        self.model = build_q_network()
        self.target_model = build_q_network()
        self.update_target_model()

        self.batch_size = config.BATCH_SIZE
        self.optimizer = self.model.optimizer      # AdamW
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.global_replay_memory = deque(maxlen=config.MEMORY_SIZE)

        # Pre-allocated buffer để giảm allocate lại
        self.batch_size = config.BATCH_SIZE

    def reset_global_replay_memory(self):
        self.global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    
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
        q_values = self.model([s1, s2, s3], training=False)[0].numpy()
        
        # Apply mask
        if valid_actions_mask is not None:
            q_values = np.where(valid_actions_mask, q_values, -np.inf)
        
        return np.argmax(q_values)
    
    @tf.function
    def train_step(self, state_batch, action_batch, reward_batch,
                   next_state_batch, done_batch):
        """
        state_batch       : [3 tensor]
        next_state_batch  : [3 tensor]
        action_batch      : int32 tensor
        reward_batch      : float32
        """

        # ------------------------------------------
        # Compute target Q-values
        # ------------------------------------------
        next_q = self.target_model(next_state_batch, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)

        targets = reward_batch + (1.0 - tf.cast(done_batch, tf.float32)) * \
                  config.GAMMA * max_next_q

        # ------------------------------------------
        # Train main network
        # ------------------------------------------
        with tf.GradientTape() as tape:
            q_values = self.model(state_batch, training=True)

            # Gather Q(s,a)
            batch_indices = tf.range(self.batch_size, dtype=tf.int32)
            action_indices = tf.stack([batch_indices, action_batch], axis=1)
            predicted = tf.gather_nd(q_values, action_indices)

            loss = self.loss_fn(targets, predicted)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    # -------------------------------------------------
    #  Wrapper train — chuẩn bị batch rồi gọi graph
    # -------------------------------------------------
    def train(self):
        if len(self.global_replay_memory) < self.batch_size:
            return 0.0

        idx = np.random.choice(len(self.global_replay_memory), self.batch_size, replace=False)

        # Prebuild numpy arrays
        state1 = []
        state2 = []
        state3 = []

        next1 = []
        next2 = []
        next3 = []

        actions = []
        rewards = []
        dones = []

        for i in idx:
            s, a, r, ns, d = self.global_replay_memory[i]
            state1.append(s[0])
            state2.append(s[1])
            state3.append(s[2])

            next1.append(ns[0])
            next2.append(ns[1])
            next3.append(ns[2])

            actions.append(a)
            rewards.append(r)
            dones.append(d)

        # Convert to TensorFlow
        state_batch = [
            tf.convert_to_tensor(np.array(state1), dtype=tf.float32),
            tf.convert_to_tensor(np.array(state2), dtype=tf.float32),
            tf.convert_to_tensor(np.array(state3), dtype=tf.float32),
        ]

        next_state_batch = [
            tf.convert_to_tensor(np.array(next1), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next2), dtype=tf.float32),
            tf.convert_to_tensor(np.array(next3), dtype=tf.float32),
        ]

        loss = self.train_step(
            state_batch=state_batch,
            action_batch=tf.convert_to_tensor(actions, dtype=tf.int32),
            reward_batch=tf.convert_to_tensor(rewards, dtype=tf.float32),
            next_state_batch=next_state_batch,
            done_batch=tf.convert_to_tensor(dones, dtype=tf.bool),
        )

        return float(loss.numpy())