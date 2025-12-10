# runners/train.py
import os
import numpy as np
from collections import deque

import config
from environment.gym_env import Env
from agent.agent import Agent
from runners.utils import run_single_episode, plot_training_results

def main():
    print("\n" + "="*80)
    print("STARTING DRL TRAINING FOR SFC PROVISIONING")
    print("="*80)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    # Initialize environment & agent
    env = Env()
    agent = Agent()
    
    # Training state
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    epsilon = config.EPSILON_START
    
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    
    # Buffers for current update
    current_update_rewards = []
    current_update_ars = []
    
    print(f"\nConfiguration:")
    print(f"  - Total Updates: {config.TRAIN_UPDATES}")
    print(f"  - Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print(f"  - Actions per Time Step: {config.ACTIONS_PER_TIME_STEP}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Memory Size: {config.MEMORY_SIZE}")
    print(f"  - Epsilon: {config.EPSILON_START} → {config.EPSILON_MIN} (decay={config.EPSILON_DECAY})")
    print("="*80)
    
    # Main training loop
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        print(f"\n[UPDATE {update_idx}/{config.TRAIN_UPDATES}]")
        
        # Run episodes
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            # Run episode
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon, training_mode=True)
            
            # Store transitions
            global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            # Update epsilon
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
            
            # Progress
            print(f"  Ep {ep_idx+1:3d}: Reward={reward:7.1f}  |  AR={ar:5.1f}%  |  ε={epsilon:.4f}", 
                  end="\r", flush=True)
        
        print()  # New line after episodes
        
        # Train network
        print(f"  Training network...", end=" ", flush=True)
        loss = agent.train(global_replay_memory)
        agent.update_target_model()
        print(f"Loss={loss:.4f}")
        
        # Calculate averages
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        # Reset buffers
        current_update_rewards = []
        current_update_ars = []
        
        print(f"  → Avg Reward: {avg_reward:.1f}  |  Avg AR: {avg_ar:.2f}%")
        
        # Save best model
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_{config.WEIGHTS_FILE}')
            print(f"  ★ New best model saved! (AR={best_ar:.2f}%)")
        
        # Save checkpoint every 50 updates
        if update_idx % 50 == 0:
            agent.model.save_weights(f'models/checkpoint_{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved at update {update_idx}")
    
    # Save final model
    agent.model.save_weights(f'models/{config.WEIGHTS_FILE}')
    
    # Plot results
    plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png')
    
    # Final statistics
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total Episodes: {len(all_ars)}")
    print(f"Final Avg AR (last 100 eps): {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best AR achieved: {best_ar:.2f}%")
    print(f"Final Epsilon: {epsilon:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()