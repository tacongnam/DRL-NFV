import numpy as np
import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN
from collections import deque
import os
import matplotlib.pyplot as plt

def main():
    # Khởi tạo môi trường và Agent
    env = SFCNVEnv()
    agent = SFC_DQN()
    
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    epsilon = config.EPSILON_START
    
    print(f"=== Starting SFC Provisioning Training ===")
    print(f"Total Updates: {config.TRAIN_UPDATES}")
    print(f"Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print(f"Total Episodes: {config.TRAIN_UPDATES * config.EPISODES_PER_UPDATE}")
    print(f"\nTraining Flow:")
    print(f"  - Run {config.EPISODES_PER_UPDATE} episodes consecutively")
    print(f"  - After {config.EPISODES_PER_UPDATE} episodes → Train model (1 update)")
    print(f"  - Repeat {config.TRAIN_UPDATES} times")
    print("=" * 80)

    total_episodes_run = 0
    best_ar = 0.0
    
    # Tracking metrics
    all_rewards = []
    all_ars = []
    
    for update_cnt in range(1, config.TRAIN_UPDATES + 1):
        
        episode_rewards = []
        episode_ars = []
        
        # Chạy E episodes, sau đó mới update 1 lần (Paper: U updates, E episodes per update)
        for ep_in_update in range(1, config.EPISODES_PER_UPDATE + 1):
            state, _ = env.reset() # Random DC count
            action_mask = env._get_valid_actions_mask()  # Get initial mask
            episode_memory = []
            total_reward = 0
            done = False
            
            while not done:
                # Use action masking to prevent invalid actions
                action = agent.get_action(state, epsilon, valid_actions_mask=action_mask)
                next_state, reward, done, _, info = env.step(action)
                
                # Get next action mask
                action_mask = info.get('action_mask', None)

                episode_memory.append((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
            
            # Đưa vào Replay Memory sau mỗi episode
            global_replay_memory.extend(episode_memory)

            episode_rewards.append(total_reward)
            episode_ars.append(info['acc_ratio'])
            total_episodes_run += 1

            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
                epsilon = max(config.EPSILON_MIN, epsilon)
            
            # In progress trong 1 update
            if ep_in_update % 1 == 0 or ep_in_update == config.EPISODES_PER_UPDATE:
                print(f"\r   [Update {update_cnt}/{config.TRAIN_UPDATES}] "
                    f"Episode {ep_in_update}/{config.EPISODES_PER_UPDATE}: "
                    f"Reward={total_reward:.1f}, AR={info['acc_ratio']:.2f}%, "
                    f"DCs={env.num_active_dcs}, Epsilon={epsilon:.3f}", end="")

        # SAU KHI CHẠY HẾT E EPISODES → TRAINING 1 LẦN
        print()  # Xuống dòng
        loss = agent.train(global_replay_memory)
        agent.update_target_model()
        
        avg_rew = np.mean(episode_rewards)
        avg_ar = np.mean(episode_ars)
        
        # Track overall progress
        all_rewards.extend(episode_rewards)
        all_ars.extend(episode_ars)
        
        # Track best model
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights('models/best_' + config.WEIGHTS_FILE)
            print(f"   >>> Best AR so far: {best_ar:.2f}% - saved! <<<")
        
        print(f"[Update {update_cnt}/{config.TRAIN_UPDATES} COMPLETED] "
              f"Avg Reward: {avg_rew:.2f} | "
              f"Avg AR: {avg_ar:.2f}% | "
              f"Loss: {loss:.4f} | Epsilon: {epsilon:.4f}")
        
        # Show overall trend every 5 updates
        if update_cnt % 5 == 0 or update_cnt == config.TRAIN_UPDATES:
            recent_50 = all_ars[-50:] if len(all_ars) >= 50 else all_ars
            recent_50_rew = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
            print(f"   [Trend] Last 50 eps: AR={np.mean(recent_50):.2f}%, Reward={np.mean(recent_50_rew):.2f}")
        
        print("-" * 80)

    # Lưu Model cuối cùng
    agent.model.save_weights(f'models/{config.WEIGHTS_FILE}')
    
    # Vẽ biểu đồ training progress
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Acceptance Ratio over time
        episodes = list(range(1, len(all_ars) + 1))
        ax1.plot(episodes, all_ars, alpha=0.3, label='Per Episode')
        
        # Moving average (window=20)
        window = 20
        if len(all_ars) >= window:
            moving_avg = np.convolve(all_ars, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(all_ars) + 1), moving_avg, 
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Acceptance Ratio (%)')
        ax1.set_title('Training Progress: Acceptance Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward over time
        ax2.plot(episodes, all_rewards, alpha=0.3, label='Per Episode')
        
        if len(all_rewards) >= window:
            moving_avg_rew = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(all_rewards) + 1), moving_avg_rew,
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Training Progress: Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fig/training_progress.png', dpi=150)
        print(f"Training progress plot saved to: training_progress.png")
    except Exception as e:
        print(f"Could not create training plot: {e}")
    
    print(f"\n{'='*80}")
    print(f"Training Completed!")
    print(f"Total Episodes Run: {total_episodes_run}")
    print(f"Overall Avg AR: {np.mean(all_ars):.2f}%")
    print(f"Overall Avg Reward: {np.mean(all_rewards):.2f}")
    print(f"Final model saved to: models/{config.WEIGHTS_FILE}")
    print(f"Best model saved to: models/best_{config.WEIGHTS_FILE} (AR={best_ar:.2f}%)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()