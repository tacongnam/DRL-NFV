# DRL/runners/eval_vae.py
import sys
import os
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.vae_env import VAEEnv
from agents.dqn_agent import Agent
from agents.vae_agent import VAEModel
from envs.observer import Observer
from runners.experiments import run_experiment_performance, run_experiment_scalability

def main():
    print("\n" + "="*80)
    print("EVALUATING GenAI-DRL MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Load GenAI Model
    state_dim = Observer.get_state_dim()
    vae_model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"Loading GenAI: {genai_path}")
        vae_model.load_weights(genai_path)
        
        norm_path = f'{genai_path}_norm_stats.npz'
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            vae_model.set_normalization_params(
                mean=float(data['mean']),
                std=float(data['std'])
            )
    else:
        print(f"\n✗ GenAI not found at {genai_path}!")
        return
    
    # Load data
    data_file = 'data_1_9/cogent_centers_easy_s1.json'
    
    if os.path.exists(data_file):
        print(f"Loading data from: {data_file}")
        
        from data_info.read_data import Read_data
        reader = Read_data(data_file)
        
        graph = reader.get_G()
        dc_list = reader.get_V()
        vnf_specs = reader.get_F()
        requests_data = reader.get_R()
        
        config.update_vnf_specs(vnf_specs)
        
        env = VAEEnv(vae_model=vae_model, data_collection_mode=False,
                     graph=graph, dcs=dc_list, requests_data=requests_data)
    else:
        print(f"Warning: Data file not found: {data_file}")
        env = VAEEnv(vae_model=vae_model, data_collection_mode=False)
    
    agent = Agent()
    
    # Load DRL Weights
    weights_path = f'models/best_genai_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/genai_{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"Loading DRL: {weights_path}")
        dummy_state, _ = env.reset()
        agent.get_action(dummy_state, 0.0)
        agent.model.load_weights(weights_path)
        print("✓ Models loaded successfully")
    else:
        print(f"\n✗ DRL weights not found: {weights_path}")
        return
    
    # Run Experiments
    run_experiment_performance(
        env=env, 
        agent=agent, 
        episodes=config.TEST_EPISODES, 
        file_prefix="genai_"
    )
    
    run_experiment_scalability(
        env=env, 
        agent=agent, 
        episodes=config.TEST_EPISODES, 
        file_prefix="genai_"
    )
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETED in {total_time/60:.1f} min")
    print("Check 'fig/genai_result_*.png'")
    print("="*80)

if __name__ == "__main__":
    main()