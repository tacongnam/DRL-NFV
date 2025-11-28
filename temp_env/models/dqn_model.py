import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from config import TRAINING_CONFIG, VNF_TYPES

class DQNModel:
    def __init__(self, dc_state_dim, sfc_state_dim, network_state_dim):
        self.dc_state_dim = dc_state_dim
        self.sfc_state_dim = sfc_state_dim
        self.network_state_dim = network_state_dim
        self.num_actions = 2 * len(VNF_TYPES) + 1
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = keras.optimizers.Adam(TRAINING_CONFIG['lr_dqn'])
        self.epsilon = TRAINING_CONFIG['epsilon_start']
        
    def _build_model(self):
        dc_input = layers.Input(shape=(self.dc_state_dim,), name='dc_input')
        sfc_input = layers.Input(shape=(self.sfc_state_dim,), name='sfc_input')
        network_input = layers.Input(shape=(self.network_state_dim,), name='network_input')
        
        dc_branch = layers.Dense(64, activation='relu')(dc_input)
        dc_branch = layers.Dense(32, activation='relu')(dc_branch)
        
        sfc_branch = layers.Dense(64, activation='relu')(sfc_input)
        sfc_branch = layers.Dense(32, activation='relu')(sfc_branch)
        
        network_branch = layers.Dense(128, activation='relu')(network_input)
        network_branch = layers.Dense(64, activation='relu')(network_branch)
        
        concat = layers.Concatenate()([dc_branch, sfc_branch, network_branch])
        
        x = layers.Dense(128, activation='relu')(concat)
        attention = layers.Dense(128, activation='softmax')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.Dense(64, activation='relu')(x)
        q_values = layers.Dense(self.num_actions)(x)
        
        return keras.Model(inputs=[dc_input, sfc_input, network_input], 
                          outputs=q_values, name='dqn')
    
    def get_action(self, dc_state, sfc_state, network_state, valid_actions=None):
        if np.random.rand() < self.epsilon:
            if valid_actions is not None:
                return np.random.choice(valid_actions)
            return np.random.randint(self.num_actions)
        
        dc_state = np.expand_dims(dc_state, 0)
        sfc_state = np.expand_dims(sfc_state, 0)
        network_state = np.expand_dims(network_state, 0)
        
        q_values = self.model([dc_state, sfc_state, network_state])[0].numpy()
        
        if valid_actions is not None:
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q)
        
        return np.argmax(q_values)
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def decay_epsilon(self):
        self.epsilon = max(TRAINING_CONFIG['epsilon_end'], 
                          self.epsilon * TRAINING_CONFIG['epsilon_decay'])
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        dc_states, sfc_states, network_states = states
        next_dc_states, next_sfc_states, next_network_states = next_states
        
        with tf.GradientTape() as tape:
            q_values = self.model([dc_states, sfc_states, network_states])
            next_q_values = self.target_model([next_dc_states, next_sfc_states, next_network_states])
            
            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q = rewards + (1 - dones) * TRAINING_CONFIG['gamma'] * max_next_q
            
            loss = tf.reduce_mean(tf.square(target_q - q_values))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def save_weights(self, path):
        self.model.save_weights(f"{path}_dqn.weights.h5")
        
    def load_weights(self, path):
        self.model.load_weights(f"{path}_dqn.weights.h5")
        self.update_target_model()