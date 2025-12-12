import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import TRAINING_CONFIG

class VAEModel:
    def __init__(self, state_dim, latent_dim=TRAINING_CONFIG['vae_latent_dim']):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.value_network = self._build_value_network()
        self.optimizer = keras.optimizers.Adam(TRAINING_CONFIG['lr_vae'])
        
    def _build_encoder(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        return keras.Model(inputs, [z_mean, z_log_var], name='encoder')
    
    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.state_dim)(x)
        return keras.Model(latent_inputs, outputs, name='decoder')
    
    def _build_value_network(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.Dense(32, activation='relu')(x)
        value = layers.Dense(1)(x)
        return keras.Model(latent_inputs, value, name='value_network')
    
    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def encode(self, state):
        z_mean, z_log_var = self.encoder(state)
        z = self.sampling(z_mean, z_log_var)
        return z, z_mean, z_log_var
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_dc_values(self, dc_states):
        dc_states = np.array(dc_states)
        z, _, _ = self.encode(dc_states)
        values = self.value_network(z)
        return values.numpy().flatten()
    
    @tf.function
    def train_step(self, current_states, next_states, values):
        current_states = tf.cast(current_states, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        values = tf.cast(values, tf.float32)
        
        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(current_states)
            reconstructed = self.decode(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(next_states - reconstructed), axis=1)
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            predicted_values = self.value_network(z)
            value_loss = tf.reduce_mean(tf.square(values - predicted_values))
            
            total_loss = reconstruction_loss + 0.1 * kl_loss + value_loss
        
        trainable_vars = (self.encoder.trainable_variables + 
                         self.decoder.trainable_variables + 
                         self.value_network.trainable_variables)
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return total_loss, reconstruction_loss, kl_loss, value_loss
    
    def save_weights(self, path):
        self.encoder.save_weights(f"{path}_encoder.weights.h5")
        self.decoder.save_weights(f"{path}_decoder.weights.h5")
        self.value_network.save_weights(f"{path}_value.weights.h5")
        
    def load_weights(self, path):
        self.encoder.load_weights(f"{path}_encoder.weights.h5")
        self.decoder.load_weights(f"{path}_decoder.weights.h5")
        self.value_network.load_weights(f"{path}_value.weights.h5")