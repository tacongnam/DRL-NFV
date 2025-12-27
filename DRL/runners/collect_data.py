import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DRL import config
from DRL.envs.vae_env import VAEEnv
from DRL.agents.dqn_agent import Agent
from DRL.envs.vae_trainer import VAETrainer

def collect_data():
    """Optimized data collection"""    
    print("\n" + "="*80)
    print("DATA COLLECTION FOR GenAI")
    print("="*80)
    
    # Load DRL
    agent = Agent()

    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
         weights_path = f'models/{config.WEIGHTS_FILE}'

    if not os.path.exists(weights_path):
        print(f"\n✗ No DRL weights found!\nRun: python scripts.py train")
        return
    
    print(f"Loading DRL: {weights_path}")
    env = VAEEnv(vae_model=None, data_collection_mode=True)
    state, _ = env.reset()
    agent.model.load_weights(weights_path)  
    print("✓ DRL loaded")
    
    # Initialize trainer
    trainer = VAETrainer()
    
    print(f"\nCollecting from {config.GENAI_DATA_EPISODES} episodes (random DC, trained DRL)")
    print(f"Sample interval: every {config.GENAI_SAMPLE_INTERVAL} steps")
    
    for ep in range(config.GENAI_DATA_EPISODES):
        state, _ = env.reset(num_dcs=np.random.randint(3, 7))
        done = False
        ep_samples = 0
        
        while not done:
            mask = env._get_valid_actions_mask()
            action = agent.get_action(state, epsilon=0.05, valid_actions_mask=mask)
            
            state, _, done, _, _ = env.step(action)
            env.count_step += 1
            
            if env.count_step % config.GENAI_SAMPLE_INTERVAL == 0:
                transitions = env.get_dc_transitions()
                for dc_id, prev_state, curr_state, value in transitions:
                    trainer.collect_transition(prev_state, curr_state, value)
                    ep_samples += 1
        
        # End of episode
        transitions = env.get_dc_transitions()
        for dc_id, prev_state, curr_state, value in transitions:
            trainer.collect_transition(
                prev_state,
                curr_state,
                value
            )
            ep_samples += 1
        
        stats = trainer.get_dataset_stats()
        
        # Log after each episode
        print(f"Ep {ep+1:2d}/{config.GENAI_DATA_EPISODES}: "
              f"samples={ep_samples:3d} | "
              f"total={stats['vae_samples']:5d} | ")
    
    final_stats = trainer.get_dataset_stats()
    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETED")
    print(f"{'='*80}")
    print(f"  VAE samples: {final_stats['vae_samples']}")
    print(f"  Value samples: {final_stats['value_samples']}")
    
    return trainer

def train_vae_model(trainer):
    """Train GenAI"""
    print(f"\n{'='*80}")
    print("TRAINING GenAI MODEL")
    print(f"{'='*80}")
    
    trainer.train_vae()
    trainer.train_value_network()
    
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/genai_model')
    
    print(f"\n{'='*80}")
    print(f"GenAI TRAINING COMPLETED")
    print(f"{'='*80}")

def main():
    # Collect
    trainer = collect_data()
    
    if trainer.get_dataset_stats()['vae_samples'] < 500:
        print(f"\n⚠ Dataset too small, consider more episodes")
        return
    
    # Train
    train_vae_model(trainer)
    
    print(f"\n{'='*80}")
    print("\nNext: python scripts.py train --mode genai")

if __name__ == "__main__":
    main()