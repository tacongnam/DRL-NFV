import tensorflow as tf
from keras import layers, Model
from config import *

class GenAIAgent(Model):
    def __init__(self):
        super(GenAIAgent, self).__init__()
        
        # --- Encoder ---
        self.enc_in = layers.Input(shape=(STATE_DIM_DC,))
        h = layers.Dense(64, activation='relu')(self.enc_in)
        h = layers.Dense(32, activation='relu')(h)
        self.z_mean = layers.Dense(LATENT_DIM)(h)
        self.z_log_var = layers.Dense(LATENT_DIM)(h)
        
        # Sampling
        def sampler(args):
            z_m, z_lv = args
            eps = tf.random.normal(shape=tf.shape(z_m))
            return z_m + tf.exp(0.5 * z_lv) * eps
            
        self.z = layers.Lambda(sampler)([self.z_mean, self.z_log_var])
        self.encoder = Model(self.enc_in, [self.z_mean, self.z_log_var, self.z], name="Encoder")
        
        # --- Decoder ---
        self.lat_in = layers.Input(shape=(LATENT_DIM,))
        h = layers.Dense(32, activation='relu')(self.lat_in)
        h = layers.Dense(64, activation='relu')(h)
        self.dec_out = layers.Dense(STATE_DIM_DC, activation='sigmoid')(h) # Normalized states
        self.decoder = Model(self.lat_in, self.dec_out, name="Decoder")
        
        # --- Value Network ---
        self.val_in = layers.Input(shape=(LATENT_DIM,))
        h = layers.Dense(32, activation='relu')(self.val_in)
        h = layers.Dense(16, activation='relu')(h)
        self.val_out = layers.Dense(1, activation='linear')(h)
        self.value_net = Model(self.val_in, self.val_out, name="ValueNet")
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR_VAE)

    @tf.function
    def train_step(self, x, x_next, values):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            x_recon = self.decoder(z)
            v_pred = self.value_net(z)
            
            # VAE Loss
            mse = tf.reduce_mean(tf.keras.losses.mse(x_next, x_recon))
            kl = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            
            # Value Loss
            val_loss = tf.reduce_mean(tf.keras.losses.mse(values, v_pred))
            
            total_loss = mse + 0.1 * kl + val_loss
            
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss

    def predict_value(self, states):
        _, _, z = self.encoder(states)
        return self.value_net(z)