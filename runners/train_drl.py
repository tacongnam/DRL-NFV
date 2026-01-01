# DRL/runners/train_drl.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from envs.drl_env import DRLEnv
from agents.dqn_agent import Agent
from runners.core import train_model_common

def main():
    """Train DRL agent with loaded data"""
    
    # Load data from file
    data_file = 'data_1_9/cogent_centers_easy_s1.json'
    
    if os.path.exists(data_file):
        print(f"Loading data from: {data_file}")
        
        from data_info.read_data import Read_data
        reader = Read_data(data_file)
        
        # Get components
        graph = reader.get_G()
        dc_list = reader.get_V()
        vnf_specs = reader.get_F()
        requests_data = reader.get_R()
        
        # Update global VNF specs
        config.update_vnf_specs(vnf_specs)
        
        print(f"  Servers: {reader.get_num_servers()}")
        print(f"  Total nodes: {graph.number_of_nodes()}")
        print(f"  Links: {graph.number_of_edges()}")
        print(f"  VNF types: {len(vnf_specs)}")
        print(f"  Requests: {len(requests_data)}")
        
        # Create environment with loaded data
        env = DRLEnv(graph=graph, dcs=dc_list, requests_data=requests_data)
    else:
        print(f"Warning: Data file not found: {data_file}")
        print("Using fallback random topology")
        env = DRLEnv()
    
    # Setup agent
    agent = Agent()
    
    # Train
    train_model_common(
        env=env,
        agent=agent,
        title="Standard DRL (Priority-based)",
        save_prefix="" 
    )

if __name__ == "__main__":
    main()