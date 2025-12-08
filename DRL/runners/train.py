import sys
import os
import numpy as np
from collections import deque

# --- CONFIG PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.gym_env import Env
from agent.agent import Agent
from utils import run_single_episode, plot_training_results

def main():
    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)

    # Initialize Env & Agent
    env = Env()
    agent = Agent()
    
    # Training state
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    epsilon = config.EPSILON_START
    
    all_rewards = []
    all_ars = []   
    
    # buffers for current update block
    current_update_rewards = []
    current_update_ars = []
    
    best_ar = 0.0
    total_episodes_run = 0

    print(f"=== STARTING TRAINING ===")
    print(f"Total Updates: {config.TRAIN_UPDATES}, Episodes/Update: {config.EPISODES_PER_UPDATE}")
    
    # --- MAIN TRAINING LOOP ---
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        
        # 1. Collect Data (Experiences)
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            total_episodes_run += 1
            
            # Run one episode using utility function
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon, training_mode=True)
            
            # Store experience
            global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            # Decay Epsilon
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
            
            # Log progress
            print(f"\r[Upd {update_idx}] Ep {ep_idx+1}/{config.EPISODES_PER_UPDATE}: "
                  f"R={reward:.1f}, AR={ar:.1f}%, Eps={epsilon:.3f}", end="")

        # 2. Train the Model
        print(f"\n   >> Training Network (Update {update_idx})...", end="")
        loss = agent.train(global_replay_memory)
        agent.update_target_model()
        
        # 3. Stats & Logging
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        # Reset buffers for next update
        current_update_rewards = []
        current_update_ars = []
        
        print(f"\r   >> Completed Update {update_idx}: "
              f"Avg Reward={avg_reward:.1f}, Avg AR={avg_ar:.2f}%, Loss={loss:.4f}")
        
        # 4. Save Best Model
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_{config.WEIGHTS_FILE}')
            print(f"   >> New Best Model Saved (AR={best_ar:.2f}%)")

    # --- END TRAINING ---
    agent.model.save_weights(f'models/{config.WEIGHTS_FILE}')
    
    # Plot results
    plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png')
    
    print(f"\n{'='*80}")
    print(f"Training Completed. Final AR Avg: {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best Model AR: {best_ar:.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()