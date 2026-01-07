import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers
from collections import deque
import config

class DQNAgent:
    def __init__(self, state_shapes, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.state_shapes = state_shapes
        
        self.q_network = self._build_network(state_shapes, action_size)
        self.target_network = self._build_network(state_shapes, action_size)
        self.update_target_network()
        
        self.optimizer = optimizers.AdamW(
            learning_rate=config.LEARNING_RATE, 
            weight_decay=1e-4
        )
    
    def _build_network(self, state_shapes, action_size):
        input_dc = layers.Input(shape=state_shapes[0], name="dc_state")
        input_dc_demand = layers.Input(shape=state_shapes[1], name="dc_demand")
        input_global = layers.Input(shape=state_shapes[2], name="global_state")
        
        x1 = layers.Dense(32, activation='relu')(input_dc)
        x2 = layers.Dense(64, activation='relu')(input_dc_demand)
        x3 = layers.Dense(64, activation='relu')(input_global)
        
        concat = layers.Concatenate()([x1, x2, x3])
        attn = layers.Dense(concat.shape[-1], activation='sigmoid')(concat)
        x = layers.Multiply()([concat, attn])
        
        x = layers.Dense(96, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        
        q_values = layers.Dense(action_size, activation='linear')(x)
        
        return models.Model(
            inputs=[input_dc, input_dc_demand, input_global],
            outputs=q_values
        )
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state, epsilon, valid_mask=None):
        if np.random.rand() < epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            return np.random.randint(self.action_size)
        
        inputs = [np.expand_dims(s, 0) for s in state]
        
        q_values = self.q_network(inputs, training=False)[0].numpy()
        
        if valid_mask is not None:
            q_values = np.where(valid_mask, q_values, -np.inf)
        
        return np.argmax(q_values)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=None):
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        if len(self.memory) < batch_size:
            return 0.0
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = [[], [], []]
        next_states = [[], [], []]
        actions, rewards, dones = [], [], []
        
        for s, a, r, ns, d in batch:
            for i in range(3):
                states[i].append(s[i])
                next_states[i].append(ns[i])
            actions.append(a)
            rewards.append(r)
            dones.append(d)
        
        try:
            states_t = [tf.convert_to_tensor(np.array(x), dtype=tf.float32) for x in states]
            next_states_t = [tf.convert_to_tensor(np.array(x), dtype=tf.float32) for x in next_states]
            actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
            dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)
        except ValueError as e:
            print(f"Warning: Skipping batch due to shape mismatch: {e}")
            return 0.0
        
        loss = self._train_step(states_t, actions_t, rewards_t, next_states_t, dones_t)
        return float(loss.numpy())
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q = self.target_network(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + (1.0 - dones) * config.GAMMA * max_next_q
        
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            
            batch_indices = tf.range(tf.shape(actions)[0])
            action_indices = tf.stack([batch_indices, actions], axis=1)
            predicted_q = tf.gather_nd(q_values, action_indices)
            
            loss = tf.reduce_mean(tf.square(targets - predicted_q))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss
    
    def save(self, path):
        self.q_network.save_weights(f"{path}_q.weights.h5")
        self.target_network.save_weights(f"{path}_target.weights.h5")
    
    def load(self, path):
        self.q_network.load_weights(f"{path}_q.weights.h5")
        self.target_network.load_weights(f"{path}_target.weights.h5")