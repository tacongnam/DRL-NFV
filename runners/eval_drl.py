# DRL/runners/eval_drl.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.drl_env import DRLEnv
from agents.dqn_agent import Agent
from runners.experiments import run_experiment_performance, run_experiment_scalability

def main():
    print("\n" + "="*80)
    print("EVALUATING STANDARD DRL MODEL")
    print("="*80)
    
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
        
        env = DRLEnv(graph=graph, dcs=dc_list, requests_data=requests_data)
    else:
        print(f"Warning: Data file not found: {data_file}")
        env = DRLEnv()
    
    agent = Agent()
    
    # Load DRL weights
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
    
    # Run experiments
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