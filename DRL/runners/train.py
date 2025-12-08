import os
import numpy as np
from collections import deque

import config
from environment.gym_env import Env
from agent.agent import Agent
from runners.utils import run_single_episode, plot_training_results

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("fig", exist_ok=True)

    # Khởi tạo môi trường & agent
    env = Env()
    agent = Agent()
    
    # Trạng thái huấn luyện
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    epsilon = config.EPSILON_START
    
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    total_episodes_run = 0
    
    # Buffers cho mỗi update
    current_update_rewards = []
    current_update_ars = []

    print(f"=== STARTING TRAINING ===")
    print(f"Total Updates: {config.TRAIN_UPDATES}, Episodes/Update: {config.EPISODES_PER_UPDATE}")
    
    # Vòng lặp huấn luyện chính
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        
        # 1. Chạy các episode
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            total_episodes_run += 1
            
            # Chạy 1 episode
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon, training_mode=True)
            
            # Lưu vào bộ nhớ replay
            global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            # Cập nhật epsilon
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
            
            # In tiến độ
            print(f"\r[Upd {update_idx}] Ep {ep_idx+1}/{config.EPISODES_PER_UPDATE}: "
                  f"R={reward:.1f}, AR={ar:.1f}%, Eps={epsilon:.3f}", end="")

        # 2. Cập nhật mạng neural
        print(f"\n   >> Training Network (Update {update_idx})...", end="")
        loss = agent.train(global_replay_memory)
        agent.update_target_model()
        
        # 3. Tính toán và in kết quả trung bình
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        # Reset buffers
        current_update_rewards = []
        current_update_ars = []
        
        print(f"\r   >> Completed Update {update_idx}: "
              f"Avg Reward={avg_reward:.1f}, Avg AR={avg_ar:.2f}%, Loss={loss:.4f}")
        
        # 4. Lưu mô hình tốt nhất
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_{config.WEIGHTS_FILE}')
            print(f"   >> New Best Model Saved (AR={best_ar:.2f}%)")

    # Lưu mô hình cuối cùng
    agent.model.save_weights(f'models/{config.WEIGHTS_FILE}')
    
    # Vẽ đồ thị huấn luyện
    plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png')
    
    print(f"\n{'='*80}")
    print(f"Training Completed. Final AR Avg: {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best Model AR: {best_ar:.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()