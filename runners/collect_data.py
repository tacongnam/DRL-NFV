# DRL/runners/train_vae.py
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.vae_env import VAEEnv
from agents.dqn_agent import Agent
from agents.vae_agent import VAEModel
from envs.observer import Observer
from runners.core import train_model_common

def main():
    """Train GenAI-assisted DRL with loaded data"""
    
    # Load GenAI model
    state_dim = Observer.get_state_dim()
    vae_model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"Loading GenAI Model from {genai_path}...")
        vae_model.load_weights(genai_path)

        norm_path = f'{genai_path}_norm_stats.npz'
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            vae_model.set_normalization_params(
                mean=float(data['mean']),
                std=float(data['std'])
            )
        else:
            print("⚠️  Warning: Normalization stats not found")
    else:
        print(f"Error: GenAI weights not found at {genai_path}!")
        print("Please run: python scripts.py collect first.")
        return

    # Load data from file
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
        
        print(f"  Servers: {reader.get_num_servers()}")
        print(f"  Requests: {len(requests_data)}")
        
        env = VAEEnv(vae_model=vae_model, data_collection_mode=False,
                     graph=graph, dcs=dc_list, requests_data=requests_data)
    else:
        print(f"Warning: Data file not found: {data_file}")
        env = VAEEnv(vae_model=vae_model, data_collection_mode=False)
    
    # Setup agent
    agent = Agent()
    
    # Train
    train_model_common(
        env=env,
        agent=agent,
        title="GenAI-Assisted DRL",
        save_prefix="genai_" 
    )

if __name__ == "__main__":
    main()