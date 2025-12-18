import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, models, optimizers

class VAEEncoder(models.Model):
    """Encoder: DC_State → Latent z (Optimized but more expressive)"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Improved architecture: 2 hidden layers for better representation
        # Still lighter than model.py but more expressive than before
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
        """Inference: only mean (deterministic encoding)"""
        z_mean, _ = self(inputs, training=False)
        return z_mean


class VAEDecoder(models.Model):
    """Decoder: Latent z → Next_DC_State (Optimized but more expressive)"""
    
    def __init__(self, output_dim):
        super().__init__()
        # Mirror encoder architecture for symmetry
        self.dense1 = layers.Dense(32, activation='relu', name='dec_fc1')
        self.dense2 = layers.Dense(48, activation='relu', name='dec_fc2')
        self.output_layer = layers.Dense(output_dim, activation='linear', name='dec_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return self.output_layer(x)


class ValueNetwork(models.Model):
    """Value Network: Latent z → Scalar (Improved with 2 layers)"""
    
    def __init__(self):
        super().__init__()
        # Added one more layer for better value approximation
        self.dense1 = layers.Dense(24, activation='relu', name='val_fc1')
        self.dense2 = layers.Dense(12, activation='relu', name='val_fc2')
        self.output_layer = layers.Dense(1, activation='linear', name='val_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return tf.squeeze(self.output_layer(x), axis=-1)


class VAEModel:
    """Complete GenAI module (Optimized & Corrected)"""
    
    def __init__(self, state_dim, latent_dim=16):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(state_dim)
        self.value_net = ValueNetwork()
        
        # Optimized learning rates
        self.vae_optimizer = optimizers.Adam(learning_rate=0.001)
        self.value_optimizer = optimizers.Adam(learning_rate=0.0005)
        
        # Build models
        dummy = tf.zeros((1, state_dim))
        self._build_models(dummy)
    
    def _build_models(self, dummy_input):
        """Initialize all model weights"""
        z_mean, z_log_var = self.encoder(dummy_input)
        _ = self.decoder(z_mean)
        _ = self.value_net(z_mean)
    
    def sampling(self, z_mean, z_log_var):
        """Reparameterization trick for VAE sampling"""
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def train_vae_step(self, current_states, next_states):
        """
        Train VAE to predict next_states given current_states
        CRITICAL: Decoder reconstructs NEXT state, not current state
        This aligns with the paper's methodology
        """
        with tf.GradientTape() as tape:
            # Encode current state
            z_mean, z_log_var = self.encoder(current_states, training=True)
            z = self.sampling(z_mean, z_log_var)
            
            # Decode to predict NEXT state (after DRL action)
            reconstructed = self.decoder(z, training=True)
            
            # Reconstruction loss: how well we predict next_state
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(next_states - reconstructed), axis=1)
            )
            
            # KL divergence loss: regularization term
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Total loss with beta=0.1 (as per paper, not 0.05)
            total_loss = recon_loss + 0.1 * kl_loss
        
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.vae_optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return total_loss, recon_loss, kl_loss
    
    @tf.function
    def train_value_step(self, current_states, target_values):
        """
        Train Value Network on embedded representations
        Encoder is frozen during this phase
        """
        # Encode with frozen encoder
        z_mean, _ = self.encoder(current_states, training=False)
        
        with tf.GradientTape() as tape:
            predicted_values = self.value_net(z_mean, training=True)
            loss = tf.reduce_mean(tf.square(target_values - predicted_values))
        
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))
        
        return loss
    
    def predict_dc_values(self, dc_states):
        """
        Vectorized inference: predict values for all DCs
        
        Args:
            dc_states: array shape (num_dcs, state_dim)
        
        Returns:
            values: array shape (num_dcs,) - importance score for each DC
        """
        dc_states_tf = tf.convert_to_tensor(dc_states, dtype=tf.float32)
        z_mean = self.encoder.encode(dc_states_tf)
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