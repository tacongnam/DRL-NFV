import numpy as np
import tensorflow as tf
import config
from agents.vae_model import VAEModel
from envs.observer import Observer

class VAETrainer:
    """Optimized training pipeline"""
    
    def __init__(self):
        state_dim = Observer.get_state_dim()
        self.model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
        
        # Smaller buffers
        self.max_size = config.GENAI_MEMORY_SIZE
        self.ptr = 0
        self.size = 0
        
        # Buffer cho VAE: (current_state, next_state)
        self.vae_curr_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.vae_next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        
        # Buffer cho Value: (state, value)
        self.val_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.val_values = np.zeros((self.max_size,), dtype=np.float32)
    
    def collect_transition(self, prev_state, curr_state, value):
        # Ghi đè theo vòng tròn (Circular Buffer)
        idx = self.ptr
        
        self.vae_curr_states[idx] = prev_state
        self.vae_next_states[idx] = curr_state
        
        self.val_states[idx] = prev_state
        self.val_values[idx] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def train_vae(self, epochs=None, batch_size=None):
        if epochs is None:
            epochs = config.GENAI_VAE_EPOCHS
        if batch_size is None:
            batch_size = config.GENAI_BATCH_SIZE
            
        if self.size < batch_size:
            print(f"Not enough VAE data: {self.size}")
            return
        
        print(f"\nTraining VAE: {self.size} samples, {epochs} epochs")
        
        # Slicing trực tiếp từ Numpy Buffer
        curr_data = self.vae_curr_states[:self.size]
        next_data = self.vae_next_states[:self.size]
        
        dataset = tf.data.Dataset.from_tensor_slices((curr_data, next_data))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_current, batch_next in dataset:
                loss, _, _ = self.model.train_vae_step(batch_current, batch_next)
                total_loss += loss.numpy()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/num_batches:.4f}")
        
        print(f"✓ VAE training completed")
    
    def train_value_network(self, epochs=None, batch_size=None):
        if epochs is None:
            epochs = config.GENAI_VALUE_EPOCHS
        if batch_size is None:
            batch_size = config.GENAI_BATCH_SIZE
            
        # SỬA LỖI: Dùng self.size thay vì len(dataset)
        if self.size < batch_size:
            print(f"Not enough Value data: {self.size}")
            return
        
        print(f"\nTraining Value Network: {self.size} samples, {epochs} epochs")
        
        # SỬA LỖI: Slicing trực tiếp từ Numpy Buffer thay vì list comprehension
        states = self.val_states[:self.size]
        values = self.val_values[:self.size]
        
        dataset = tf.data.Dataset.from_tensor_slices((states, values))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_values in dataset:
                loss = self.model.train_value_step(batch_states, batch_values)
                total_loss += loss.numpy()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/num_batches:.4f}")
        
        print(f"✓ Value Network training completed")
    
    def save_model(self, path='models/genai_model'):
        self.model.save_weights(path)
        print(f"✓ Saved to {path}")
    
    def load_model(self, path='models/genai_model'):
        self.model.load_weights(path)
        print(f"✓ Loaded from {path}")
    
    def get_dataset_stats(self):
        # SỬA LỖI: Trả về self.size
        return {
            'vae_samples': self.size,
            'value_samples': self.size
        }