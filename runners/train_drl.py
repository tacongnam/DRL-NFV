import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.drl_env import DRLEnv
from agents.dqn_agent import Agent
from runners.core import train_model_common
from read_data import Read_data

def main():
    # Load data from file
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
    print(f"  Total nodes: {graph.number_of_nodes()}")
    print(f"  Links: {graph.number_of_edges()}")
    print(f"  VNF types: {config.NUM_VNF_TYPES}")
    print(f"  Requests: {len(requests_data)}")
    print(f"  Action space: {config.ACTION_SPACE_SIZE}")
    
    env = DRLEnv(graph=graph, dcs=dc_list, requests_data=requests_data)
    agent = Agent()
    
    train_model_common(
        env=env,
        agent=agent,
        title="Standard DRL (Priority-based)",
        save_prefix=""
    )

if __name__ == "__main__":
    main()