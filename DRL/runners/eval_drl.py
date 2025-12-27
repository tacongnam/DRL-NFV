# runners/eval_drl.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DRL import config
from envs.drl_env import Env
from agents.dqn_agent import Agent
from runners.experiments import run_experiment_performance, run_experiment_scalability

def main():
    print("\n" + "="*80)
    print("EVALUATING STANDARD DRL MODEL")
    print("="*80)
    
    env = Env()
    agent = Agent()
    
    # Load DRL weights (Standard version)
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
         weights_path = f'models/{config.WEIGHTS_FILE}'

    if os.path.exists(weights_path):
        print(f"Loading weights: {weights_path}")
        dummy_state, _ = env.reset()
        agent.get_action(dummy_state, 0.0)
        agent.model.load_weights(weights_path)
    else:
        print(f"✗ Weights not found: {weights_path}")
        return
    
    # Run Experiments (No prefix -> default names)
    run_experiment_performance(
        env, agent, 
        episodes=config.TEST_EPISODES, 
        file_prefix="" 
    )
    
    run_experiment_scalability(
        env, agent, 
        episodes=config.TEST_EPISODES, 
        file_prefix=""
    )

    print("\nCheck 'fig/result_*.png'")

if __name__ == "__main__":
    main()