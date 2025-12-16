import numpy as np
import tensorflow as tf
from collections import deque
import config
from genai.model import GenAIModel
from genai.observer import DCStateObserver

class GenAITrainer:
    """Training pipeline cho GenAI model"""
    
    def __init__(self):
        state_dim = DCStateObserver.get_state_dim()
        self.model = GenAIModel(state_dim, latent_dim=32)
        
        # Dataset buffers
        self.vae_dataset = deque(maxlen=50000)
        # Format: (current_state, next_state)
        
        self.value_dataset = deque(maxlen=50000)
        # Format: (current_state, value)
    
    def collect_transition(self, dc, sfc_manager, dc_prev_state):
        """
        Thu thập 1 transition sau khi DRL thực hiện action
        
        Args:
            dc: DataCenter object
            sfc_manager: SFC_Manager object
            dc_prev_state: Previous state của DC (before action)
        """
        # Current state (after action)
        current_state = DCStateObserver.get_dc_state(dc, sfc_manager)
        
        # Calculate value
        value = DCStateObserver.calculate_dc_value(dc, sfc_manager, dc_prev_state)
        
        # Store
        self.vae_dataset.append((dc_prev_state, current_state))
        self.value_dataset.append((dc_prev_state, value))
    
    def train_vae(self, epochs=50, batch_size=64):
        """Train VAE"""
        if len(self.vae_dataset) < batch_size:
            print(f"Not enough data for VAE training: {len(self.vae_dataset)}")
            return
        
        print(f"\n{'='*60}")
        print(f"Training VAE with {len(self.vae_dataset)} samples")
        print(f"{'='*60}")
        
        # Convert to arrays
        data = list(self.vae_dataset)
        current_states = np.array([d[0] for d in data], dtype=np.float32)
        next_states = np.array([d[1] for d in data], dtype=np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((current_states, next_states))
        dataset = dataset.shuffle(10000).batch(batch_size)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            num_batches = 0
            
            for batch_current, batch_next in dataset:
                loss, recon, kl = self.model.train_vae_step(batch_current, batch_next)
                total_loss += loss.numpy()
                total_recon += recon.numpy()
                total_kl += kl.numpy()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_kl = total_kl / num_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f} | "
                      f"Recon={avg_recon:.4f} | "
                      f"KL={avg_kl:.4f}")
        
        print(f"✓ VAE training completed")
    
    def train_value_network(self, epochs=30, batch_size=64):
        """Train Value Network (Encoder frozen)"""
        if len(self.value_dataset) < batch_size:
            print(f"Not enough data for Value Network training: {len(self.value_dataset)}")
            return
        
        print(f"\n{'='*60}")
        print(f"Training Value Network with {len(self.value_dataset)} samples")
        print(f"{'='*60}")
        
        # Convert to arrays
        data = list(self.value_dataset)
        states = np.array([d[0] for d in data], dtype=np.float32)
        values = np.array([d[1] for d in data], dtype=np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((states, values))
        dataset = dataset.shuffle(10000).batch(batch_size)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_values in dataset:
                loss = self.model.train_value_step(batch_states, batch_values)
                total_loss += loss.numpy()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print(f"✓ Value Network training completed")
    
    def save_model(self, path='models/genai_model'):
        """Save GenAI model"""
        self.model.save_weights(path)
        print(f"✓ GenAI model saved to {path}")
    
    def load_model(self, path='models/genai_model'):
        """Load GenAI model"""
        self.model.load_weights(path)
        print(f"✓ GenAI model loaded from {path}")
    
    def get_dataset_stats(self):
        """Get statistics về dataset"""
        return {
            'vae_samples': len(self.vae_dataset),
            'value_samples': len(self.value_dataset)
        }