import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras import layers, models

class VAEEncoder(models.Model):
    """Encoder: DC_State → Latent z"""
    
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 2 hidden layers for better representation
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
    """Decoder: Latent z → Next_DC_State"""
    
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
    """Value Network: Latent z → Scalar value"""
    
    def __init__(self):
        super().__init__()
        # 2 layers for better value approximation
        self.dense1 = layers.Dense(24, activation='relu', name='val_fc1')
        self.dense2 = layers.Dense(12, activation='relu', name='val_fc2')
        self.output_layer = layers.Dense(1, activation='linear', name='val_output')
    
    def call(self, z, training=False):
        x = self.dense1(z)
        x = self.dense2(x)
        return tf.squeeze(self.output_layer(x), axis=-1)