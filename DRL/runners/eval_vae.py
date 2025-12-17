# runners/eval_vae.py
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.vae_env import VAEEnv
from agents.dqn_agent import Agent
from agents.vae_model import VAEModel # Hoặc VAEModel
from envs.observer import Observer

# Import các hàm experiment đã refactor
from runners.experiments import run_experiment_performance, run_experiment_scalability

def main():
    print("\n" + "="*80)
    print("EVALUATING GenAI-DRL MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load GenAI Model
    state_dim = Observer.get_state_dim()
    genai_model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"Loading GenAI: {genai_path}")
        genai_model.load_weights(genai_path)
    else:
        print(f"\n✗ GenAI not found at {genai_path}!")
        return
    
    # 2. Setup Env
    env = VAEEnv(genai_model=genai_model, data_collection_mode=False)
    agent = Agent()
    
    # 3. Load DRL Weights (Best GenAI version)
    weights_path = f'models/best_genai_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/genai_{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"Loading DRL: {weights_path}")
        # Dummy forward pass để build model
        dummy_state, _ = env.reset()
        agent.get_action(dummy_state, 0.0)
        agent.model.load_weights(weights_path)
        print("✓ Models loaded successfully")
    else:
        print(f"\n✗ DRL weights not found: {weights_path}")
        return
    
    # 4. Run Experiments
    # Exp 1: Performance
    run_experiment_performance(
        env=env, 
        agent=agent, 
        episodes=config.TEST_EPISODES, 
        file_prefix="genai_" # Prefix để phân biệt file ảnh
    )
    
    # Exp 2: Scalability
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