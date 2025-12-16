import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.genai_env import GenAIEnv
from agent.agent import Agent
from genai.model import GenAIModel
from genai.observer import DCStateObserver
from runners.utils import plot_training_results

def run_single_episode(env, agent, epsilon, training_mode=True):
    """Run episode với GenAI DC selection"""
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0.0
    done = False
    episode_memory = []
    
    while not done:
        # Adaptive epsilon
        epsilon_curr = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                       np.exp(-env.count_step * 3 / config.DECAY_STEP)
        
        action = agent.get_action(state, epsilon_curr, valid_actions_mask=action_mask)
        
        next_state, reward, done, _, info = env.step(action)
        next_action_mask = info.get('action_mask', None)
        
        if training_mode:
            episode_memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        action_mask = next_action_mask
        
        env.count_step += 1
        
        # Train periodically
        if training_mode and env.count_step % config.TARGET_NETWORK_UPDATE == 0:
            print()
            print(f"  Training network...", end=" ", flush=True)
            loss = agent.train()
            agent.update_target_model()
            print(f"Loss={loss:.4f}")
    
    acc_ratio = info.get('acceptance_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio

def main():
    print("\n" + "="*80)
    print("TRAINING GenAI-DRL FOR SFC PROVISIONING")
    print("="*80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    # Load GenAI model
    state_dim = DCStateObserver.get_state_dim()
    genai_model = GenAIModel(state_dim, latent_dim=32)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"\nLoading GenAI model from: {genai_path}")
        genai_model.load_weights(genai_path)
        print("✓ GenAI model loaded")
    else:
        print(f"\n✗ GenAI model not found!")
        print("Please run: python runners/collect_data.py")
        return
    
    # Initialize environment with GenAI
    env = GenAIEnv(genai_model=genai_model, data_collection_mode=False)
    agent = Agent()
    
    # Training state
    epsilon = config.EPSILON_START
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    
    current_update_rewards = []
    current_update_ars = []
    
    print(f"\nConfiguration:")
    print(f"  - Mode: GenAI-DRL (VAE-based DC Selection)")
    print(f"  - Total Updates: {config.TRAIN_UPDATES}")
    print(f"  - Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print("="*80)
    
    # Main training loop
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        print(f"\n[UPDATE {update_idx}/{config.TRAIN_UPDATES}]")
        
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon, training_mode=True)
            
            episode_epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                             np.exp(-env.count_step * 3 / config.DECAY_STEP)
            
            agent.global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
            
            print(f"  Ep {ep_idx+1:3d}: Reward={reward:7.1f}  |  AR={ar:5.1f}%  |  "
                  f"ε={episode_epsilon:.4f} | step={env.count_step}", 
                  end="\r", flush=True)
        
        print()
        
        # Calculate averages
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        current_update_rewards = []
        current_update_ars = []
        
        print(f"  → Avg Reward: {avg_reward:.1f}  |  Avg AR: {avg_ar:.2f}%")
        
        # Save best model
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_genai_{config.WEIGHTS_FILE}')
            print(f"  ★ New best GenAI-DRL model! (AR={best_ar:.2f}%)")
        
        # Checkpoint
        if update_idx % 50 == 0:
            agent.model.save_weights(f'models/checkpoint_genai_{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved")
    
    # Save final
    agent.model.save_weights(f'models/genai_{config.WEIGHTS_FILE}')
    
    # Plot
    plot_training_results(all_rewards, all_ars, save_path='fig/training_genai_progress.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total Episodes: {len(all_ars)}")
    print(f"Final Avg AR (last 100): {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best AR: {best_ar:.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()