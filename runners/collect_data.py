import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.vae_env import VAEEnv
from agents.dqn_agent import Agent
from envs.vae_trainer import VAETrainer
from envs.observer import Observer

def main():
    print("\n" + "="*80)
    print("DATA COLLECTION FOR GENAI")
    print("="*80)
    
    # Load data
    data_file = 'data_1_9/cogent_centers_easy_s1.json'
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return
    
    from runners.read_data import Read_data
    reader = Read_data(data_file)
    
    graph = reader.get_G()
    dc_list = reader.get_V()
    vnf_specs = reader.get_F()
    requests_data = reader.get_R()
    
    config.update_vnf_specs(vnf_specs)
    
    print(f"  Servers: {reader.get_num_servers()}")
    print(f"  Requests: {len(requests_data)}")
    
    # Environment in data collection mode (random DC selection)
    env = VAEEnv(vae_model=None, data_collection_mode=True,
                 graph=graph, dcs=dc_list, requests_data=requests_data)
    
    # Load pre-trained DQN
    agent = Agent()
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"Loading DQN: {weights_path}")
        dummy_state, _ = env.reset()
        agent.get_action(dummy_state, 0.0)
        agent.model.load_weights(weights_path)
    else:
        print(f"Warning: DQN weights not found at {weights_path}")
        print("Proceeding with random policy...")
    
    # Initialize trainer
    trainer = VAETrainer()
    
    # Collect data
    print(f"\nCollecting data: {config.GENAI_DATA_EPISODES} episodes")
    
    for ep in range(config.GENAI_DATA_EPISODES):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = agent.get_action(state, epsilon=0.0, 
                                     valid_actions_mask=env._get_valid_actions_mask())
            next_state, _, done, _, info = env.step(action)
            
            # Collect transitions at intervals
            if step_count % config.GENAI_SAMPLE_INTERVAL == 0:
                transitions = env.get_dc_transitions()
                for dc_id, prev_state, curr_state, value in transitions:
                    trainer.collect_transition(prev_state, curr_state, value)
            
            state = next_state
            step_count += 1
        
        stats = trainer.get_dataset_stats()
        print(f"  Ep {ep+1}/{config.GENAI_DATA_EPISODES}: Collected {stats['vae_samples']} samples", 
              end="\r", flush=True)
    
    print(f"\n\nDataset: {trainer.get_dataset_stats()}")
    
    # Train VAE
    trainer.train_vae()
    
    # Normalize values and train Value Network
    values = trainer.val_values[:trainer.size]
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    print(f"\nValue normalization: mean={mean_val:.2f}, std={std_val:.2f}")
    trainer.val_values[:trainer.size] = (values - mean_val) / (std_val + 1e-8)
    
    trainer.train_value_network()
    
    # Save
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/genai_model')
    
    # Save normalization stats
    np.savez('models/genai_model_norm_stats.npz', mean=mean_val, std=std_val)
    print(f"✓ Saved normalization stats")
    
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()