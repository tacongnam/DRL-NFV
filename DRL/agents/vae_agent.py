import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, models, optimizers
from DRL.agents.vae_model import VAEEncoder, VAEDecoder, ValueNetwork

class VAEModel:
    """Complete GenAI module with denormalization support"""
    
    def __init__(self, state_dim, latent_dim=16):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(state_dim)
        self.value_net = ValueNetwork()
        
        # Optimized learning rates
        self.vae_optimizer = optimizers.Adam(learning_rate=0.001)
        self.value_optimizer = optimizers.Adam(learning_rate=0.0005)
        
        # Normalization parameters
        self.value_mean = 0.0
        self.value_std = 1.0
        
        # Build models
        dummy = tf.zeros((1, state_dim))
        self._build_models(dummy)
    
    def _build_models(self, dummy_input):
        """Initialize all model weights"""
        z_mean, z_log_var = self.encoder(dummy_input)
        _ = self.decoder(z_mean)
        _ = self.value_net(z_mean)
    
    def set_normalization_params(self, mean, std):
        """Set normalization parameters for denormalization"""
        self.value_mean = float(mean)
        self.value_std = float(std)
        print(f"✓ Set normalization params: mean={mean:.2f}, std={std:.2f}")
    
    def sampling(self, z_mean, z_log_var):
        """Reparameterization trick for VAE sampling"""
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def train_vae_step(self, current_states, next_states):
        """Train VAE to predict next_states given current_states"""
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
            
            total_loss = recon_loss + 0.1 * kl_loss
        
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.vae_optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return total_loss, recon_loss, kl_loss
    
    @tf.function
    def train_value_step(self, current_states, target_values):
        """Train Value Network on embedded representations"""
        z_mean, _ = self.encoder(current_states, training=False)
        
        with tf.GradientTape() as tape:
            predicted_values = self.value_net(z_mean, training=True)
            loss = tf.reduce_mean(tf.square(target_values - predicted_values))
        
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        
        return loss
    
    def predict_dc_values(self, dc_states):
        """
        Vectorized inference with denormalization
        
        Args:
            dc_states: array shape (num_dcs, state_dim)
        
        Returns:
            values: array shape (num_dcs,) - denormalized importance scores
        """
        dc_states_tf = tf.convert_to_tensor(dc_states, dtype=tf.float32)
        z_mean = self.encoder.encode(dc_states_tf)
        normalized_values = self.value_net(z_mean, training=False)
        
        # Denormalize predictions
        denormalized_values = (
            normalized_values.numpy() * self.value_std + self.value_mean
        )
        
        return denormalized_values
    
    def save_weights(self, path_prefix):
        """Save all model weights"""
        self.encoder.save_weights(f'{path_prefix}_encoder.weights.h5')
        self.decoder.save_weights(f'{path_prefix}_decoder.weights.h5')
        self.value_net.save_weights(f'{path_prefix}_value.weights.h5')
    
    def load_weights(self, path_prefix):
        """Load all model weights"""
        self.encoder.load_weights(f'{path_prefix}_encoder.weights.h5')
        self.decoder.load_weights(f'{path_prefix}_decoder.weights.h5')
        self.value_net.load_weights(f'{path_prefix}_value.weights.h5')