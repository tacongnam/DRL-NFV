import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.drl_env import DRLEnv
from agents.dqn_agent import Agent
from runners.experiments import run_experiment_overall, run_experiment_scalability
from read_data import Read_data

def main():
    print("\n" + "="*80)
    print("EVALUATING STANDARD DRL MODEL")
    print("="*80)
    
    # Load data
    data_file = 'data/test.json'
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return

    reader = Read_data(data_file)
    
    graph = reader.get_G()
    dc_list = reader.get_V()
    vnf_specs = reader.get_F()
    requests_data = reader.get_R()
    
    config.update_vnf_specs(vnf_specs)
    config.ACTION_SPACE_SIZE = config.get_action_space_size()
    
    print(f"  Servers: {reader.get_num_servers()}")
    print(f"  VNF types: {config.NUM_VNF_TYPES}")
    print(f"  Requests: {len(requests_data)}")
    print(f"  Action space: {config.ACTION_SPACE_SIZE}")
    
    env = DRLEnv(graph=graph, dcs=dc_list, requests_data=requests_data)
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
    run_experiment_overall(env, agent, episodes=config.TEST_EPISODES, file_prefix="")
    
    # Scalability (if multiple DC configs supported)
    # run_experiment_scalability(env, agent, dc_configs=config.TEST_NUM_DCS_RANGE, 
    #                            episodes=config.TEST_EPISODES, file_prefix="")

    print("\nCheck 'fig/result_*.png'")

if __name__ == "__main__":
    main()