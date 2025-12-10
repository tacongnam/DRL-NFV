# runners/evaluate.py
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.gym_env import Env
from agent.agent import Agent
from runners.utils import run_experiment_performance, run_experiment_scalability

def main():
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    # Initialize
    env = Env()
    agent = Agent()
    
    # Load weights
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"\nLoading weights from: {weights_path}")
        try:
            # Build model architecture by doing a dummy prediction
            dummy_state, _ = env.reset()
            agent.get_action(dummy_state, 0.0)
            
            agent.model.load_weights(weights_path)
            print("✓ Weights loaded successfully")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return
    else:
        print(f"\n✗ No weights found at: {weights_path}")
        print("Please run train.py first to train the model.")
        return
    
    # Run experiments
    print(f"\nTest Configuration:")
    print(f"  - Episodes per experiment: {config.TEST_EPISODES}")
    print(f"  - Epsilon (exploration): {config.TEST_EPSILON}")
    print(f"  - DC configurations: {config.TEST_FIG3_DCS}")
    
    # Experiment 1: Performance per SFC type
    run_experiment_performance(env, agent, episodes=config.TEST_EPISODES)
    
    # Experiment 2: Scalability
    run_experiment_scalability(env, agent, episodes=config.TEST_EPISODES)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print("Check the 'fig/' directory for generated plots.")

if __name__ == "__main__":
    main()