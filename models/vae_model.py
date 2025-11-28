import tensorflow as tf
from keras import layers, Model
import config

class VAEWithValueHead(Model):
    def __init__(self, input_dim):
        super(VAEWithValueHead, self).__init__()
        self.latent_dim = config.LATENT_DIM
        
        # Encoder: Input -> Latent Z
        self.encoder_inputs = layers.InputLayer(input_shape=(input_dim,))
        self.enc_dense1 = layers.Dense(64, activation="relu")
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")
        
        # Decoder: Latent Z -> NEXT STATE (x')
        # [Paper Fig. 1]: Output is x'=d(z), representing DC Next State
        self.dec_dense1 = layers.Dense(64, activation="relu")
        self.dec_output = layers.Dense(input_dim, activation="sigmoid") 
        
        # Value Network: Latent Z -> Value
        self.val_dense1 = layers.Dense(32, activation="relu")
        self.val_output = layers.Dense(1, activation="linear") 

    def encode(self, x):
        x = self.enc_dense1(x)
        return self.z_mean(x), self.z_log_var(x)

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var * 0.5) * eps

    def decode(self, z):
        x = self.dec_dense1(z)
        return self.dec_output(x)
    
    def predict_value(self, z):
        v = self.val_dense1(z)
        return self.val_output(v)

    def call(self, inputs):
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        
        # Quan trọng: Decoder output là dự đoán Next State
        next_state_pred = self.decode(z)
        value_pred = self.predict_value(z)
        
        return next_state_pred, value_pred, mean, log_var
    
    def get_dc_score(self, dc_state):
        dc_state = tf.convert_to_tensor([dc_state], dtype=tf.float32)
        mean, _ = self.encode(dc_state)
        val = self.predict_value(mean)
        return val.numpy()[0][0]