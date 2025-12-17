"""
Online GenAI-DRL Training

Train DRL and GenAI together in single phase
Saves ~40% time compared to offline approach
"""

import os
import sys
import numpy as np
import time
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.gym_env import Env
from agent.agent import Agent
from genai.model import GenAIModel
from genai.observer import DCStateObserver
from genai.online_buffer import DCTransitionBuffer
from runners.utils import plot_training_results

class OnlineGenAIDRLTrainer:
    """Online trainer for GenAI + DRL"""
    
    def __init__(self):
        # Environment
        self.env = Env()
        
        # DRL Agent
        self.agent = Agent()
        
        # GenAI Model
        state_dim = DCStateObserver.get_state_dim()
        self.genai_model = GenAIModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
        
        # Buffers
        self.genai_buffer = DCTransitionBuffer(capacity=config.GENAI_BUFFER_SIZE)
        
        # Training state
        self.step_count = 0
        self.use_genai = False
        self.genai_weight = 0.0
        
        # Statistics
        self.genai_update_count = 0
        self.drl_update_count = 0
        
    def select_dc(self):
        """Select DC with progressive strategy"""
        
        # Phase 1: Warmup (random)
        if self.step_count < config.GENAI_WARMUP_STEPS:
            return np.random.randint(0, len(self.env.dcs))
        
        # Phase 2: Not trained yet
        if not self.use_genai:
            return np.random.randint(0, len(self.env.dcs))
        
        # Phase 3: Progressive GenAI
        if np.random.random() < self.genai_weight:
            # Use GenAI
            dc_states = DCStateObserver.get_all_dc_states(
                self.env.dcs, 
                self.env.sfc_manager
            )
            values = self.genai_model.predict_dc_values(dc_states)
            return np.argmax(values)
        else:
            # Random for exploration
            return np.random.randint(0, len(self.env.dcs))
    
    def update_genai_weight(self):
        """Update progressive weight"""
        if self.step_count < config.GENAI_WARMUP_STEPS:
            self.genai_weight = 0.0
        else:
            progress = (self.step_count - config.GENAI_WARMUP_STEPS) / config.GENAI_PROGRESSIVE_STEPS
            self.genai_weight = min(1.0, max(0.0, progress))
    
    def train_genai(self):
        """Train GenAI with mini-epochs"""
        
        if len(self.genai_buffer) < config.GENAI_MIN_BUFFER:
            print(f"  Skipping GenAI (buffer={len(self.genai_buffer)} < {config.GENAI_MIN_BUFFER})")
            return
        
        print(f"\n[GenAI Update #{self.genai_update_count + 1}]")
        print(f"  Buffer: {len(self.genai_buffer)} samples")
        
        batch_size = config.GENAI_BATCH_SIZE
        num_batches = len(self.genai_buffer) // batch_size
        
        # Mini-epochs
        for epoch in range(config.GENAI_MINI_EPOCHS):
            indices = np.random.permutation(len(self.genai_buffer))
            total_vae_loss = 0.0
            total_value_loss = 0.0
            
            for i in range(num_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]
                batch = self.genai_buffer.sample(batch_idx)
                
                # Train VAE
                vae_loss, _, _ = self.genai_model.train_vae_step(
                    tf.convert_to_tensor(batch['current']),
                    tf.convert_to_tensor(batch['next'])
                )
                
                # Train Value
                value_loss = self.genai_model.train_value_step(
                    tf.convert_to_tensor(batch['current']),
                    tf.convert_to_tensor(batch['value'])
                )
                
                total_vae_loss += vae_loss.numpy()
                total_value_loss += value_loss.numpy()
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{config.GENAI_MINI_EPOCHS}: "
                      f"VAE={total_vae_loss/num_batches:.4f}, "
                      f"Value={total_value_loss/num_batches:.4f}")
        
        self.genai_update_count += 1
        self.use_genai = True
        print(f"  ✓ GenAI updated (total updates: {self.genai_update_count})")
    
    def train_episode(self, epsilon):
        """Train one episode"""
        
        state, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        
        while not done:
            # Store DC states before action
            dc_prev_states = [
                DCStateObserver.get_dc_state(dc, self.env.sfc_manager)
                for dc in self.env.dcs
            ]
            
            # Select DC
            dc_id = self.select_dc()
            curr_dc = self.env.dcs[dc_id]
            
            # Get state and action
            from environment.observer import Observer
            dc_full_state = Observer.get_full_state(curr_dc, self.env.sfc_manager)
            
            from environment.utils import get_valid_actions_mask
            action_mask = get_valid_actions_mask(curr_dc, self.env.sfc_manager.active_requests)
            
            action = self.agent.get_action(dc_full_state, epsilon, action_mask)
            
            # Execute action
            next_state, reward, done, _, info = self.env.step(action)
            
            # Store for DRL
            self.agent.global_replay_memory.append(
                (dc_full_state, action, reward, next_state, done)
            )
            
            # Store for GenAI
            dc_next_states = [
                DCStateObserver.get_dc_state(dc, self.env.sfc_manager)
                for dc in self.env.dcs
            ]
            
            for i, dc in enumerate(self.env.dcs):
                value = DCStateObserver.calculate_dc_value(
                    dc, self.env.sfc_manager, dc_prev_states[i]
                )
                self.genai_buffer.add(dc_prev_states[i], dc_next_states[i], value)
            
            episode_reward += reward
            episode_steps += 1
            self.step_count += 1
            
            # Update DRL
            if self.step_count % config.DRL_UPDATE_INTERVAL == 0:
                loss = self.agent.train()
                self.agent.update_target_model()
                self.drl_update_count += 1
                print(f"\n  [DRL Update #{self.drl_update_count}] "
                      f"Step={self.step_count}, Loss={loss:.4f}")
            
            # Update GenAI
            if self.step_count % config.GENAI_UPDATE_INTERVAL == 0:
                self.train_genai()
                self.update_genai_weight()
                print(f"  GenAI weight: {self.genai_weight:.2f}")
            
            state = next_state
        
        stats = self.env.sfc_manager.get_statistics()
        return episode_reward, stats['acceptance_ratio'], episode_steps

def main():
    print("\n" + "="*80)
    print("ONLINE GenAI-DRL TRAINING")
    print("="*80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    trainer = OnlineGenAIDRLTrainer()
    
    print(f"\nConfiguration:")
    print(f"  - Mode: ONLINE (DRL + GenAI together)")
    print(f"  - DRL update: every {config.DRL_UPDATE_INTERVAL} steps")
    print(f"  - GenAI update: every {config.GENAI_UPDATE_INTERVAL} steps")
    print(f"  - Warmup: {config.GENAI_WARMUP_STEPS} steps (random DC)")
    print(f"  - Progressive: {config.GENAI_PROGRESSIVE_STEPS} steps (0→100% GenAI)")
    print(f"  - Mini-epochs: {config.GENAI_MINI_EPOCHS}")
    print("="*80)
    
    epsilon = config.EPSILON_START
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    
    start_time = time.time()
    
    # Training loop
    for update in range(1, config.TRAIN_UPDATES + 1):
        update_start = time.time()
        update_rewards = []
        update_ars = []
        
        for ep in range(config.EPISODES_PER_UPDATE):
            epsilon_curr = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                          np.exp(-trainer.step_count * 3 / config.DECAY_STEP)
            
            reward, ar, steps = trainer.train_episode(epsilon_curr)
            
            update_rewards.append(reward)
            update_ars.append(ar)
            
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
        
        # Update statistics
        avg_reward = np.mean(update_rewards)
        avg_ar = np.mean(update_ars)
        
        all_rewards.extend(update_rewards)
        all_ars.extend(update_ars)
        
        update_time = time.time() - update_start
        elapsed = time.time() - start_time
        
        # Log
        print(f"\nUpdate {update:2d}/{config.TRAIN_UPDATES}: "
              f"AR={avg_ar:5.1f}% | "
              f"Reward={avg_reward:6.1f} | "
              f"Steps={trainer.step_count} | "
              f"GenAI_w={trainer.genai_weight:.2f} | "
              f"time={update_time:.1f}s | "
              f"total={elapsed/60:.1f}min")
        
        # Save best
        if avg_ar > best_ar:
            best_ar = avg_ar
            trainer.agent.model.save_weights(f'models/best_online_{config.WEIGHTS_FILE}')
            trainer.genai_model.save_weights('models/online_genai_model')
            print(f"  ★ Best models saved (AR={best_ar:.2f}%)")
        
        # Checkpoint
        if update % 20 == 0:
            trainer.agent.model.save_weights(f'models/checkpoint_online_{update}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved")
    
    # Save final
    trainer.agent.model.save_weights(f'models/online_{config.WEIGHTS_FILE}')
    trainer.genai_model.save_weights('models/online_genai_model')
    
    # Plot
    plot_training_results(all_rewards, all_ars, save_path='fig/training_online_progress.png')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED in {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("="*80)
    print(f"Episodes: {len(all_ars)}")
    print(f"Total steps: {trainer.step_count}")
    print(f"DRL updates: {trainer.drl_update_count}")
    print(f"GenAI updates: {trainer.genai_update_count}")
    print(f"Final AR (last 100): {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best AR: {best_ar:.2f}%")
    print(f"GenAI weight: {trainer.genai_weight:.2f}")
    print("="*80)
    
    # Buffer stats
    buffer_stats = trainer.genai_buffer.get_stats()
    print(f"\nFinal Buffer Stats:")
    print(f"  Size: {buffer_stats['size']}")
    print(f"  Mean value: {buffer_stats['mean_value']:.2f}")
    print(f"  Std value: {buffer_stats['std_value']:.2f}")

if __name__ == "__main__":
    main()