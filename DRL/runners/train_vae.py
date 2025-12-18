# runners/train_vae.py
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
    # 1. Load Pre-trained GenAI Model
    state_dim = Observer.get_state_dim()
    vae_model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"[Info] Loading GenAI Model from {genai_path}...")
        vae_model.load_weights(genai_path)

        norm_path = f'{genai_path}_norm_stats.npz'
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            vae_model.set_normalization_params(
                mean=float(data['mean']),
                std=float(data['std'])
            )
        else:
            print("⚠️  Warning: Normalization stats not found, using defaults (mean=0, std=1)")
    else:
        print(f"[Error] GenAI weights not found at {genai_path}!")
        print("Please run: python scripts.py collect first.")
        return

    # 2. Setup Environment (GenAI Mode)
    env = VAEEnv(vae_model=vae_model, data_collection_mode=False)
    
    # 3. Setup Agent
    agent = Agent()
    
    # 4. Run Common Training Loop
    train_model_common(
        env=env,
        agent=agent,
        title="GenAI-Assisted DRL",
        save_prefix="genai_" 
    )

if __name__ == "__main__":
    main()