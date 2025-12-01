import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import sys
sys.path.append('..')
from config import *

class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    
    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W1(inputs) + self.W2(inputs)))
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = inputs * attention_weights
        return context_vector

class DQNModel:
    def __init__(self, num_vnf_types, num_sfc_types):
        self.num_vnf_types = num_vnf_types
        self.num_sfc_types = num_sfc_types
        self.num_actions = 2 * num_vnf_types + 1
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=TRAINING_CONFIG['learning_rate'])
        self.loss_fn = keras.losses.Huber()
    
    def _build_model(self):
        state1_dim = 2 * self.num_vnf_types + 2
        state2_dim = self.num_sfc_types * (1 + 2 * self.num_vnf_types)
        state3_dim = self.num_sfc_types * (4 + self.num_vnf_types)
        
        input1 = layers.Input(shape=(state1_dim,), name='state1')
        input2 = layers.Input(shape=(state2_dim,), name='state2')
        input3 = layers.Input(shape=(state3_dim,), name='state3')
        
        x1 = layers.Dense(128, activation='relu')(input1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dense(128, activation='relu')(x1)
        x1 = layers.Dense(64, activation='relu')(x1)
        
        x2 = layers.Dense(128, activation='relu')(input2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dense(128, activation='relu')(x2)
        x2 = layers.Dense(64, activation='relu')(x2)
        
        x3 = layers.Dense(128, activation='relu')(input3)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dense(128, activation='relu')(x3)
        x3 = layers.Dense(64, activation='relu')(x3)
        
        concat = layers.Concatenate()([x1, x2, x3])
        
        attention_input = layers.Dense(192, activation='relu')(concat)
        attention_output = AttentionLayer(192)(attention_input)
        
        attended_features = layers.Flatten()(attention_output)
        
        x = layers.Dense(256, activation='relu')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        value_stream = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, activation='linear', name='value')(value_stream)
        
        advantage_stream = layers.Dense(64, activation='relu')(x)
        advantages = layers.Dense(self.num_actions, activation='linear', name='advantages')(advantage_stream)
        
        mean_advantage = tf.reduce_mean(advantages, axis=1, keepdims=True)
        output = value + (advantages - mean_advantage)
        
        model = keras.Model(inputs=[input1, input2, input3], outputs=output)
        return model
    
    def predict(self, state):
        state_input = [
            np.expand_dims(state['state1'], 0),
            np.expand_dims(state['state2'], 0),
            np.expand_dims(state['state3'], 0)
        ]
        return self.model(state_input, training=False)[0].numpy()
    
    def train_on_batch(self, states, actions, rewards, next_states, dones):
        states_input = [
            np.array([s['state1'] for s in states]),
            np.array([s['state2'] for s in states]),
            np.array([s['state3'] for s in states])
        ]
        
        next_states_input = [
            np.array([s['state1'] for s in next_states]),
            np.array([s['state2'] for s in next_states]),
            np.array([s['state3'] for s in next_states])
        ]
        
        current_q_values = self.model(states_input, training=False).numpy()
        next_q_values = self.target_model(next_states_input, training=False).numpy()
        
        max_next_q_values = np.max(next_q_values, axis=1)
        
        target_q_values = current_q_values.copy()
        
        for i in range(len(states)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + TRAINING_CONFIG['gamma'] * max_next_q_values[i]
        
        with tf.GradientTape() as tape:
            q_values = self.model(states_input, training=True)
            loss = self.loss_fn(target_q_values, q_values)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        clipped_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
        
        return loss.numpy()
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        self.update_target_model()

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for idx in indices:
            s, a, r, ns, d = self.memory[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)