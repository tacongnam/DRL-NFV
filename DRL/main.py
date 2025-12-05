import numpy as np
import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN
from collections import deque
import time

def main():
    env = SFCNVEnv()
    agent = SFC_DQN()
    
    global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
    
    epsilon = config.EPSILON_START
    
    print(f"=== Starting SFC Provisioning Training ===")
    print(f"Total Updates: {config.TRAIN_UPDATES}")
    print(f"Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print(f"Actions per Time Step (1ms): {config.ACTIONS_PER_TIME_STEP}")
    print("==========================================")

    total_episodes_run = 0
    
    for update_cnt in range(1, config.TRAIN_UPDATES + 1):
        
        episode_rewards = []
        episode_ars = []
        
        for _ in range(config.EPISODES_PER_UPDATE):
            state, _ = env.reset()
            episode_memory = []
            total_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon)
                next_state, reward, done, _, info = env.step(action)

                episode_memory.append((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
            
            global_replay_memory.extend(episode_memory)

            episode_rewards.append(total_reward)
            episode_ars.append(info['acc_ratio'])
            total_episodes_run += 1
            
            print(f"\r   > Ep {total_episodes_run}: Reward={total_reward:.1f}, AR={info['acc_ratio']:.2f}%, DCs={env.num_active_dcs}", end="")

        loss = agent.train(global_replay_memory)

        if epsilon > config.EPSILON_MIN:
            epsilon *= config.EPSILON_DECAY
        
        avg_rew = np.mean(episode_rewards)
        avg_ar = np.mean(episode_ars)
        
        print(f"\n[Update {update_cnt}/{config.TRAIN_UPDATES}] "
              f"Avg Reward: {avg_rew:.2f} | "
              f"Avg Acceptance Ratio: {avg_ar:.2f}% | "
              f"Loss: {loss:.4f} | Epsilon: {epsilon:.4f}")

    # Save Final Model
    agent.model.save_weights('sfc_dqn_weights.h5')
    print("Training Completed & Model Saved.")

if __name__ == "__main__":
    main()