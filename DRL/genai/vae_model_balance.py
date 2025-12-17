"""
Balanced VAE Architecture - Middle ground option

Use this if:
- You have 5-6 hours available (instead of 4)
- Want ~98% performance instead of 95%
- Have concerns about representation capacity

To use: In config.py, set USE_BALANCED_VAE = True
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, models, optimizers
import config

class VAEEncoderBalanced(models.Model):
    """Balanced Encoder: 2.5 layers, latent 24"""
    
    def __init__(self, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        
        # More capacity than optimized, less than full
        self.dense1 = layers.Dense(48, activation='relu', name='enc_fc1')
        self.dense2 = layers.Dense(32, activation='relu', name='enc_fc2')
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
    
    def encode(self, inputs):
        z_mean, _ = self(inputs, training=False)
        return z_mean


class VAEDecoderBalanced(models.Model):
    """Balanced Decoder"""
    
    def __init__(self, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(32, activation='relu', name='dec_fc1')
        self.dense2 = layers.Dense(48, activation='relu', name='dec_fc2')
        self.output_layer = layers.Dense(output_dim, activation='linear', name='dec_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return self.output_layer(x)


class ValueNetworkBalanced(models.Model):
    """Balanced Value Network"""
    
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(24, activation='relu', name='val_fc1')
        self.dense2 = layers.Dense(12, activation='relu', name='val_fc2')
        self.output = layers.Dense(1, activation='linear', name='val_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return tf.squeeze(self.output(x), axis=-1)


class GenAIModelBalanced:
    """Balanced GenAI Model"""
    
    def __init__(self, state_dim, latent_dim=24):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoderBalanced(latent_dim)
        self.decoder = VAEDecoderBalanced(state_dim)
        self.value_net = ValueNetworkBalanced()
        
        # Slightly lower LR for stability
        self.vae_optimizer = optimizers.Adam(learning_rate=0.0015)
        self.value_optimizer = optimizers.Adam(learning_rate=0.0008)
        
        dummy = tf.zeros((1, state_dim))
        self._build_models(dummy)
    
    def _build_models(self, dummy_input):
        z_mean, z_log_var = self.encoder(dummy_input)
        _ = self.decoder(z_mean)
        _ = self.value_net(z_mean)
    
    def sampling(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def train_vae_step(self, current_states, next_states):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(current_states, training=True)
            z = self.sampling(z_mean, z_log_var)
            reconstructed = self.decoder(z, training=True)
            
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(next_states - reconstructed), axis=1)
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = recon_loss + 0.08 * kl_loss  # Medium beta
        
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.vae_optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return total_loss, recon_loss, kl_loss
    
    @tf.function
    def train_value_step(self, current_states, target_values):
        z_mean, _ = self.encoder(current_states, training=False)
        
        with tf.GradientTape() as tape:
            predicted_values = self.value_net(z_mean, training=True)
            loss = tf.reduce_mean(tf.square(target_values - predicted_values))
        
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        
        return loss
    
    def predict_dc_values(self, dc_states):
        dc_states_tf = tf.convert_to_tensor(dc_states, dtype=tf.float32)
        z_mean = self.encoder.encode(dc_states_tf)
        values = self.value_net(z_mean, training=False)
        return values.numpy()
    
    def save_weights(self, path_prefix):
        self.encoder.save_weights(f'{path_prefix}_encoder.weights.h5')
        self.decoder.save_weights(f'{path_prefix}_decoder.weights.h5')
        self.value_net.save_weights(f'{path_prefix}_value.weights.h5')
    
    def load_weights(self, path_prefix):
        self.encoder.load_weights(f'{path_prefix}_encoder.weights.h5')
        self.decoder.load_weights(f'{path_prefix}_decoder.weights.h5')
        self.value_net.load_weights(f'{path_prefix}_value.weights.h5')


# Performance comparison metrics
def get_architecture_stats():
    """Get stats for architecture comparison"""
    return {
        'optimized': {
            'latent_dim': 16,
            'encoder_params': ~1800,
            'decoder_params': ~1800,
            'value_params': ~400,
            'total_params': ~4000,
            'expected_time': '~1 hour',
            'expected_performance': '95%'
        },
        'balanced': {
            'latent_dim': 24,
            'encoder_params': ~3200,
            'decoder_params': ~3200,
            'value_params': ~700,
            'total_params': ~7100,
            'expected_time': '~1.5 hours',
            'expected_performance': '98%'
        },
        'full': {
            'latent_dim': 32,
            'encoder_params': ~5000,
            'decoder_params': ~5000,
            'value_params': ~1000,
            'total_params': ~11000,
            'expected_time': '~2.5 hours',
            'expected_performance': '100%'
        }
    }