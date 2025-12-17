import numpy as np
import tensorflow as tf
from collections import deque
import config
from genai.model import GenAIModel
from genai.observer import DCStateObserver

class GenAITrainer:
    """Optimized training pipeline"""
    
    def __init__(self):
        state_dim = DCStateObserver.get_state_dim()
        self.model = GenAIModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
        
        # Smaller buffers
        self.vae_dataset = deque(maxlen=config.GENAI_MEMORY_SIZE)
        self.value_dataset = deque(maxlen=config.GENAI_MEMORY_SIZE)
    
    def collect_transition(self, dc, sfc_manager, dc_prev_state):
        current_state = DCStateObserver.get_dc_state(dc, sfc_manager)
        value = DCStateObserver.calculate_dc_value(dc, sfc_manager, dc_prev_state)
        
        self.vae_dataset.append((dc_prev_state, current_state))
        self.value_dataset.append((dc_prev_state, value))
    
    def train_vae(self, epochs=None, batch_size=None):
        if epochs is None:
            epochs = config.GENAI_VAE_EPOCHS
        if batch_size is None:
            batch_size = config.GENAI_BATCH_SIZE
            
        if len(self.vae_dataset) < batch_size:
            print(f"Not enough VAE data: {len(self.vae_dataset)}")
            return
        
        print(f"\nTraining VAE: {len(self.vae_dataset)} samples, {epochs} epochs")
        
        data = list(self.vae_dataset)
        current_states = np.array([d[0] for d in data], dtype=np.float32)
        next_states = np.array([d[1] for d in data], dtype=np.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((current_states, next_states))
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
            
        if len(self.value_dataset) < batch_size:
            print(f"Not enough Value data: {len(self.value_dataset)}")
            return
        
        print(f"\nTraining Value Network: {len(self.value_dataset)} samples, {epochs} epochs")
        
        data = list(self.value_dataset)
        states = np.array([d[0] for d in data], dtype=np.float32)
        values = np.array([d[1] for d in data], dtype=np.float32)
        
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
        return {
            'vae_samples': len(self.vae_dataset),
            'value_samples': len(self.value_dataset)
        }