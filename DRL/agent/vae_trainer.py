import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import config
from agents import VAEAgent

class VAETrainer:
    """Quản lý Training Loop và Data Buffer cho VAE Agent."""
    
    def __init__(self, state_dim):
        # Khởi tạo Agent (The Brain)
        self.agent = VAEAgent(state_dim, latent_dim=config.GENAI_LATENT_DIM)
        
        # Cấu hình Buffer
        self.max_size = config.GENAI_MEMORY_SIZE
        self.ptr = 0
        self.size = 0
        
        # Buffer: Sử dụng Numpy array cố định để tối ưu bộ nhớ
        self.vae_curr_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.vae_next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.val_values = np.zeros((self.max_size,), dtype=np.float32)
        
        # Note: val_states chính là vae_curr_states, dùng chung để tiết kiệm RAM
    
    def collect_transition(self, prev_state, curr_state, value):
        """Lưu transition vào Circular Buffer."""
        idx = self.ptr
        
        self.vae_curr_states[idx] = prev_state
        self.vae_next_states[idx] = curr_state
        self.val_values[idx] = value
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def train_vae(self, epochs=None, batch_size=None):
        """Huấn luyện Encoder & Decoder."""
        if epochs is None: epochs = config.GENAI_VAE_EPOCHS
        if batch_size is None: batch_size = config.GENAI_BATCH_SIZE
            
        if self.size < batch_size:
            return
        
        print(f"\n>>> Training VAE ({self.size} samples)...")
        
        # Tạo Dataset từ Buffer hiện có
        curr_data = self.vae_curr_states[:self.size]
        next_data = self.vae_next_states[:self.size]
        
        dataset = tf.data.Dataset.from_tensor_slices((curr_data, next_data))
        dataset = dataset.shuffle(self.size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch_curr, batch_next in dataset:
                loss = self.agent.train_vae_step(batch_curr, batch_next)
                total_loss += loss.numpy()
                steps += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/steps:.4f}")
    
    def train_value_network(self, epochs=None, batch_size=None):
        """Huấn luyện Value Network (Latent -> Value)."""
        if epochs is None: epochs = config.GENAI_VALUE_EPOCHS
        if batch_size is None: batch_size = config.GENAI_BATCH_SIZE
            
        if self.size < batch_size:
            return
        
        print(f"\n>>> Training Value Network ({self.size} samples)...")
        
        states = self.vae_curr_states[:self.size]
        values = self.val_values[:self.size]
        
        # Cập nhật tham số chuẩn hóa cho Agent trước khi train
        mean_val = np.mean(values)
        std_val = np.std(values) + 1e-8
        self.agent.set_normalization_params(mean_val, std_val)
        
        # Normalize targets
        norm_values = (values - mean_val) / std_val
        
        dataset = tf.data.Dataset.from_tensor_slices((states, norm_values))
        dataset = dataset.shuffle(self.size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch_states, batch_vals in dataset:
                loss = self.agent.train_value_step(batch_states, batch_vals)
                total_loss += loss.numpy()
                steps += 1
                
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/steps:.4f}")

    def save_model(self, path):
        self.agent.save_weights(path)
    
    def load_model(self, path):
        self.agent.load_weights(path)