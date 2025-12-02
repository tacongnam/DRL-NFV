import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from env.sfc_environment import SFCEnvironment
from agent import DQNAgent
from config import *
import numpy as np

def quick_demo():
    print("="*60)
    print("Quick Demo: DRL-based SFC Provisioning")
    print("="*60)
    
    env = SFCEnvironment(num_dcs=4)
    
    state1_dim = 2 * len(VNF_LIST) + 2
    state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
    state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
    action_dim = 2 * len(VNF_LIST) + 1
    
    agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
    
    # Try to load a pre-trained model
    best_model_path = "checkpoints/best_model.weights.h5"
    final_model_path = "checkpoints/final_model.weights.h5"
    load_path = None
    
    if os.path.exists(best_model_path):
        load_path = best_model_path
        print(f"\n[INFO] Found best_model checkpoint. Loading from {best_model_path}...")
    elif os.path.exists(final_model_path):
        load_path = final_model_path
        print(f"\n[INFO] Found final_model checkpoint. Loading from {final_model_path}...")
    else:
        print("\n[INFO] No pre-trained checkpoint found. Training from scratch...")
    
    if load_path:
        try:
            agent.load(load_path)
            print(f"[SUCCESS] Model loaded successfully. Skipping training phase.\n")
            training_episodes = 0
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}. Proceeding with training.\n")
            training_episodes = 5
    else:
        training_episodes = 5
    
    print("Environment Setup:")
    print(f"  Number of DCs: {env.num_dcs}")
    print(f"  VNF Types: {', '.join(VNF_LIST)}")
    print(f"  SFC Types: {', '.join(SFC_TYPES.keys())}")
    print(f"  Action Space: {action_dim} (Allocate={len(VNF_LIST)}, Uninstall={len(VNF_LIST)}, Wait=1)")
    
    print(f"\nTraining for {training_episodes} episodes...")
    if training_episodes == 0:
        print("(Skipped - using pre-trained model)")
    for episode in range(training_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        # Debug counters
        action_counts = np.zeros(action_dim, dtype=int)
        invalid_count = 0
        prev_satisfied = 0
        prev_dropped = 0
        
        while not done and step < 500:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, info = env.step(action)

            # debug counters
            if 0 <= action < action_dim:
                action_counts[action] += 1
            if reward == REWARD_CONFIG['invalid_action']:
                invalid_count += 1

            # store and train
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if len(agent.memory) >= DRL_CONFIG['batch_size']:
                agent.train()

            # track per-step satisfied/dropped increments
            if 'satisfied' in info and 'dropped' in info:
                prev_satisfied = info['satisfied']
                prev_dropped = info['dropped']

            step += 1
        
        if training_episodes > 0:
            agent.decay_epsilon()
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"AccRatio={info['acceptance_ratio']:.1%}, "
              f"Satisfied={info['satisfied']}, Dropped={info['dropped']}")
        print(f"    Actions distribution: {action_counts.tolist()}")
        print(f"    Invalid actions: {invalid_count}")
    
    print("\n" + "="*60)
    print("Testing on different network configurations...")
    print("="*60)
    for num_dcs in [2, 4, 6]:
        test_env = SFCEnvironment(num_dcs=num_dcs)
        state, _ = test_env.reset()
        done = False
        step = 0
        invalid_action_count = 0
        
        while not done and step < 1000:
            action = agent.select_action(state, training=False)
            state, reward, done, _, info = test_env.step(action)
            if reward == REWARD_CONFIG['invalid_action']:
                invalid_action_count += 1
            step += 1
        
        print(f"  {num_dcs} DCs:")
        print(f"    AccRatio={info['acceptance_ratio']:.1%}, Satisfied={info['satisfied']}, Dropped={info['dropped']}")
        print(f"    Delay={info['avg_delay']:.2f}ms, ResUtil={info['resource_util']:.1%}")
        print(f"    Invalid actions: {invalid_action_count}/{step}")
        print(f"    Pending SFCs: {len(test_env.pending_sfcs)}, Active SFCs: {len(test_env.active_sfcs)}")
    
    print("\n" + "="*60)
    print("Demo completed! Run main.py for full training.")
    print("="*60)

if __name__ == "__main__":
    quick_demo()