import sys
import os

# --- CONFIG PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.gym_env import Env
from agent.agent import Agent
from runners.utils import run_experiment_performance, run_experiment_scalability

def main():
    print("=== STARTING EVALUATION ===")
    
    # 1. Init Env & Agent
    env = Env()
    agent = Agent()
    
    # 2. Load Weights
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        try:
            # Fake inference to build model shape before loading weights
            dummy_state, _ = env.reset()
            agent.get_action(dummy_state, 0.0) 
            
            agent.model.load_weights(weights_path)
            print("✓ Weights loaded.")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return
    else:
        print("Error: No weights found. Run train.py first.")
        return

    # 3. Run Experiments using Helper Functions
    run_experiment_performance(env, agent, episodes=config.TEST_EPISODES)
    run_experiment_scalability(env, agent, episodes=config.TEST_EPISODES)
    
    print("\n=== EVALUATION COMPLETED ===")

if __name__ == "__main__":
    main()