import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from config import VNF_LIST, DRL_CONFIG

class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class DQNModel:
    def __init__(self, state1_dim, state2_dim, state3_dim, action_dim):
        self.state1_dim = state1_dim
        self.state2_dim = state2_dim
        self.state3_dim = state3_dim
        self.action_dim = action_dim
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=DRL_CONFIG['learning_rate'])
    
    def _build_model(self):
        input1 = layers.Input(shape=(self.state1_dim,), name='input1')
        input2 = layers.Input(shape=(self.state2_dim,), name='input2')
        input3 = layers.Input(shape=(self.state3_dim,), name='input3')
        
        x1 = layers.Dense(128, activation='relu')(input1)
        x1 = layers.Dense(64, activation='relu')(x1)
        
        x2 = layers.Reshape((self.state2_dim // (1 + 2 * len(VNF_LIST)), 
                            1 + 2 * len(VNF_LIST)))(input2)
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(64, activation='relu')(x2)
        
        x3 = layers.Reshape((self.state3_dim // (4 + len(VNF_LIST)), 
                            4 + len(VNF_LIST)))(input3)
        x3 = layers.Dense(64, activation='relu')(x3)
        x3 = layers.Flatten()(x3)
        x3 = layers.Dense(64, activation='relu')(x3)
        
        concatenated = layers.Concatenate()([x1, x2, x3])
        
        attention_input = layers.Reshape((3, -1))(
            layers.Concatenate()([
                layers.Reshape((1, -1))(x1),
                layers.Reshape((1, -1))(x2),
                layers.Reshape((1, -1))(x3)
            ])
        )
        attended = AttentionLayer(128)(attention_input)
        
        combined = layers.Concatenate()([concatenated, attended])
        
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        
        output = layers.Dense(self.action_dim, activation='linear')(x)
        
        model = keras.Model(inputs=[input1, input2, input3], outputs=output)
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def predict(self, state1, state2, state3):
        return self.model.predict([state1, state2, state3], verbose=0)
    
    def target_predict(self, state1, state2, state3):
        return self.target_model.predict([state1, state2, state3], verbose=0)
    
    def train_on_batch(self, states1, states2, states3, actions, rewards, 
                       next_states1, next_states2, next_states3, dones):
        with tf.GradientTape() as tape:
            current_q = self.model([states1, states2, states3])
            current_q = tf.reduce_sum(current_q * tf.one_hot(actions, self.action_dim), axis=1)
            
            next_q = self.target_model([next_states1, next_states2, next_states3])
            max_next_q = tf.reduce_max(next_q, axis=1)
            
            target_q = rewards + (1 - dones) * DRL_CONFIG['gamma'] * max_next_q
            
            loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        self.update_target_model()