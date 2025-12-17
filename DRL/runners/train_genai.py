import os
import sys
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.genai_env import GenAIEnv
from agent.agent import Agent
from genai.model import GenAIModel
from genai.observer import DCStateObserver
from runners.utils import plot_training_results

def run_single_episode(env, agent, training_mode=True):
    """Run episode"""
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0.0
    done = False
    episode_memory = []
    
    while not done:
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
        
        if training_mode and env.count_step % config.TARGET_NETWORK_UPDATE == 0:
            loss = agent.train()
            agent.update_target_model()
    
    acc_ratio = info.get('acceptance_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio

def main():
    print("\n" + "="*80)
    print("TRAINING GenAI-DRL")
    print("="*80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    # Load GenAI
    state_dim = DCStateObserver.get_state_dim()
    genai_model = GenAIModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"Loading GenAI: {genai_path}")
        genai_model.load_weights(genai_path)
        print("✓ GenAI loaded")
    else:
        print(f"\n✗ GenAI not found!")
        print("Run: python runners/collect_genai_data.py")
        return
    
    env = GenAIEnv(genai_model=genai_model, data_collection_mode=False)
    agent = Agent()
    
    epsilon = config.EPSILON_START
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    
    current_update_rewards = []
    current_update_ars = []
    
    print(f"\nConfig: GenAI-DRL, {config.TRAIN_UPDATES} updates, "
          f"{config.EPISODES_PER_UPDATE} eps/update")
    print("="*80)
    
    start_time = time.time()
    
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        update_start = time.time()
        
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            reward, ar, memory_trace = run_single_episode(env, agent, training_mode=True)
            
            agent.global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
        
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        current_update_rewards = []
        current_update_ars = []
        
        update_time = time.time() - update_start
        elapsed = time.time() - start_time
        
        # Log after each update
        print(f"Update {update_idx:2d}/{config.TRAIN_UPDATES}: "
              f"AR={avg_ar:5.1f}% | "
              f"Reward={avg_reward:6.1f} | "
              f"time={update_time:.1f}s | "
              f"total={elapsed/60:.1f}min")
        
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_genai_{config.WEIGHTS_FILE}')
            print(f"  ★ Best model saved (AR={best_ar:.2f}%)")
        
        if update_idx % 20 == 0:
            agent.model.save_weights(f'models/checkpoint_genai_{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved")
    
    agent.model.save_weights(f'models/genai_{config.WEIGHTS_FILE}')
    
    plot_training_results(all_rewards, all_ars, save_path='fig/training_genai_progress.png')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED in {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("="*80)
    print(f"Episodes: {len(all_ars)}")
    print(f"Final AR (last 100): {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best AR: {best_ar:.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()