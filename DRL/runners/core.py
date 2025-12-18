import os
import time
import numpy as np
import config
# Import hàm plot từ file visualization vừa tạo
from runners.visualization import plot_training_results

def run_single_episode(env, agent, epsilon, training_mode=True):
    """
    Chạy một episode (Shared logic)
    """
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0.0
    done = False
    episode_memory = []

    decay_step = config.TRAIN_UPDATES * config.EPISODES_PER_UPDATE * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP
    
    while not done:
        if epsilon == None:
            epsilon_curr = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * np.exp(- env.count_step * 3 / decay_step)
            action = agent.get_action(state, epsilon_curr, valid_actions_mask=action_mask)
        else:
            action = agent.get_action(state, epsilon, valid_actions_mask=action_mask)
        
        next_state, reward, done, _, info = env.step(action)
        next_action_mask = info.get('action_mask', None)
        
        if training_mode:
            episode_memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        action_mask = next_action_mask
        env.count_step += 1

        if training_mode:
            if env.count_step % config.TRAIN_INTERVAL == 0:
                agent.train()
            if env.count_step % config.TARGET_NETWORK_UPDATE == 0:
                agent.update_target_model()
    
    acc_ratio = info.get('acceptance_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio

def train_agent_common(env, agent, title, save_prefix=""):
    """
    Hàm training chung
    """
    print("\n" + "="*80)
    print(f"STARTING TRAINING: {title}")
    print("="*80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    epsilon = config.EPSILON_START
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    
    current_update_rewards = []
    current_update_ars = []
    
    start_time = time.time()
    
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        update_start = time.time()
        
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon=None, training_mode=True)
            
            agent.global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
                
            print(f"  Ep {ep_idx+1:3d}: R={reward:6.1f} | AR={ar:5.1f}% | ε={epsilon:.4f}", end="\r", flush=True)
        
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        current_update_rewards = []
        current_update_ars = []
        
        update_time = time.time() - update_start
        total_time = time.time() - start_time
        
        print(f"\nUpdate {update_idx:3d}/{config.TRAIN_UPDATES}: Avg AR={avg_ar:5.2f}% | Time={update_time:.1f}s | Total={total_time/60:.1f}min")
        
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_{save_prefix}{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  ★ Best Model Saved: models/best_{save_prefix}{update_idx}_{config.WEIGHTS_FILE} (AR={best_ar:.2f}%)")
            
        if update_idx % 50 == 0:
            agent.model.save_weights(f'models/checkpoint_{save_prefix}{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved at update {update_idx}")

    final_path = f'models/{save_prefix}{config.WEIGHTS_FILE}'
    agent.model.save_weights(final_path)
    
    plot_path = f'fig/training_{save_prefix}progress.png'
    plot_training_results(all_rewards, all_ars, save_path=plot_path)
    
    print("\n" + "="*80)
    print(f"COMPLETED: {title} | Best AR: {best_ar:.2f}%")
    print("="*80)