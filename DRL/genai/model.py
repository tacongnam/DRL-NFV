import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, models, optimizers
import config

class VAEEncoder(models.Model):
    """Encoder: DC_State → Latent z"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input shape: DC state vector
        # [CPU, RAM, Storage, Installed_VNFs, Idle_VNFs, SFC_Info]
        self.dense1 = layers.Dense(64, activation='relu', name='enc_fc1')
        self.dense2 = layers.Dense(48, activation='relu', name='enc_fc2')
        
        # Latent space parameters
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
    
    def encode(self, inputs):
        """Inference: chỉ lấy mean (không sample)"""
        z_mean, _ = self(inputs, training=False)
        return z_mean


class VAEDecoder(models.Model):
    """Decoder: Latent z → Next_DC_State"""
    
    def __init__(self, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(48, activation='relu', name='dec_fc1')
        self.dense2 = layers.Dense(64, activation='relu', name='dec_fc2')
        self.output_layer = layers.Dense(output_dim, activation='linear', name='dec_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return self.output_layer(x)


class ValueNetwork(models.Model):
    """Value Network: Latent z → Scalar Value"""
    
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32, activation='relu', name='val_fc1')
        self.dense2 = layers.Dense(16, activation='relu', name='val_fc2')
        self.output_layer = layers.Dense(1, activation='linear', name='val_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return tf.squeeze(self.output_layer(x), axis=-1)  # Shape: (batch_size,)


class GenAIModel:
    """Complete GenAI module: VAE + Value Network"""
    
    def __init__(self, state_dim, latent_dim=32):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Build models
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(state_dim)
        self.value_net = ValueNetwork()
        
        # Optimizers
        self.vae_optimizer = optimizers.Adam(learning_rate=0.001)
        self.value_optimizer = optimizers.Adam(learning_rate=0.0005)
        
        # Build models with dummy data
        dummy = tf.zeros((1, state_dim))
        self._build_models(dummy)
    
    def _build_models(self, dummy_input):
        """Initialize weights"""
        z_mean, z_log_var = self.encoder(dummy_input)
        _ = self.decoder(z_mean)
        _ = self.value_net(z_mean)
    
    def sampling(self, z_mean, z_log_var):
        """Reparameterization trick"""
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def train_vae_step(self, current_states, next_states):
        """Train VAE with reconstruction + KL loss"""
        with tf.GradientTape() as tape:
            # Encode
            z_mean, z_log_var = self.encoder(current_states, training=True)
            z = self.sampling(z_mean, z_log_var)
            
            # Decode
            reconstructed = self.decoder(z, training=True)
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(next_states - reconstructed), axis=1)
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Total loss
            total_loss = recon_loss + 0.1 * kl_loss  # Beta = 0.1
        
        # Update
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.vae_optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return total_loss, recon_loss, kl_loss
    
    @tf.function
    def train_value_step(self, current_states, target_values):
        """Train Value Network (Encoder frozen)"""
        # Encode (no gradient)
        z_mean, _ = self.encoder(current_states, training=False)
        
        with tf.GradientTape() as tape:
            predicted_values = self.value_net(z_mean, training=True)
            loss = tf.reduce_mean(tf.square(target_values - predicted_values))
        
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        
        return loss
    
    def predict_dc_values(self, dc_states):
        """
        Inference: Tính value cho tất cả DC
        
        Args:
            dc_states: array shape (num_dcs, state_dim)
        
        Returns:
            values: array shape (num_dcs,)
        """
        dc_states_tf = tf.convert_to_tensor(dc_states, dtype=tf.float32)
        
        # Encode
        z_mean = self.encoder.encode(dc_states_tf)
        
        # Predict values
        values = self.value_net(z_mean, training=False)
        
        return values.numpy()
    
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