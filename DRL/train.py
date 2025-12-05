import numpy as np
import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN
from collections import deque
import os

def main():
    # Khởi tạo môi trường và Agent
    env = SFCNVEnv()
    agent = SFC_DQN()
    
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    epsilon = config.EPSILON_START
    
    print(f"=== Starting SFC Provisioning Training ===")
    print(f"Total Updates: {config.TRAIN_UPDATES}")
    print(f"Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print("==========================================")

    total_episodes_run = 0
    
    for update_cnt in range(1, config.TRAIN_UPDATES + 1):
        
        episode_rewards = []
        episode_ars = []
        
        # Loop các episodes trong 1 lần update
        for _ in range(config.EPISODES_PER_UPDATE):
            state, _ = env.reset() # Random DC count
            episode_memory = []
            total_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)

                episode_memory.append((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
            
            # Đưa vào Replay Memory
            global_replay_memory.extend(episode_memory)

            episode_rewards.append(total_reward)
            episode_ars.append(info['acc_ratio'])
            total_episodes_run += 1
            
            print(f"\r   > Ep {total_episodes_run}: Reward={total_reward:.1f}, AR={info['acc_ratio']:.2f}%, DCs={env.num_active_dcs}", end="")

        # Training Step
        loss = agent.train(global_replay_memory)

        # Decay Epsilon
        if epsilon > config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY
        
        avg_rew = np.mean(episode_rewards)
        avg_ar = np.mean(episode_ars)
        
        print(f"\n[Update {update_cnt}/{config.TRAIN_UPDATES}] "
              f"Avg Reward: {avg_rew:.2f} | "
              f"Avg AR: {avg_ar:.2f}% | "
              f"Loss: {loss:.4f} | Epsilon: {epsilon:.4f}")

    # Lưu Model
    agent.model.save_weights(config.WEIGHTS_FILE)
    print(f"Training Completed. Model saved to {config.WEIGHTS_FILE}")

if __name__ == "__main__":
    main()