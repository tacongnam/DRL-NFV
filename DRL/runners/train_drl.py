# runners/train_drl.py
import sys
import os

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DRL.envs.drl_env import DRLEnv
from DRL.agents.dqn_agent import Agent
from DRL.runners.core import train_model_common

def main():
    """
    Train DRL agent with loaded data.
    For now, creates environment without data (random topology).
    To use with data, uncomment the data loading section.
    """
    
    # Option 1: Load data from file (RECOMMENDED)
    # from data_info.read_data import Read_data
    # reader = Read_data('data_1_9/cogent_centers_easy_s1.json')
    # dc_list = reader.get_V()
    # topology = reader.get_E(len(dc_list))
    # vnf_specs = reader.get_F()
    # requests_data = reader.get_R()
    # env = DRLEnv(dcs=dc_list, topology=topology, requests_data=requests_data)
    
    # Option 2: Use random topology (FALLBACK - for testing only)
    env = DRLEnv()  # Will create random topology
    
    # 2. Setup Agent
    agent = Agent()
    
    # 3. Run Common Training Loop
    train_model_common(
        env=env,
        agent=agent,
        title="Standard DRL (Priority-based)",
        save_prefix="" 
    )

if __name__ == "__main__":
    main()