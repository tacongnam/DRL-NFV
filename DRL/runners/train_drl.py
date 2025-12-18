# runners/train_drl.py
import sys
import os

# Add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.drl_env import Env
from agents.dqn_agent import Agent
from runners.core import train_model_common

def main():
    # 1. Setup Environment
    env = Env()
    
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