import os
import sys
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.genai_env import GenAIEnv
from agent.agent import Agent
from genai.trainer import GenAITrainer

def collect_data(num_episodes=None):
    """Optimized data collection"""
    if num_episodes is None:
        num_episodes = config.GENAI_DATA_EPISODES
    
    print("\n" + "="*80)
    print("DATA COLLECTION FOR GenAI")
    print("="*80)
    
    # Load DRL
    agent = Agent()
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if not os.path.exists(weights_path):
        print(f"\n✗ No DRL weights found!")
        print("Run: python scripts.py train")
        return
    
    print(f"Loading DRL: {weights_path}")
    env = GenAIEnv(genai_model=None, data_collection_mode=True)
    state, _ = env.reset()
    agent.get_action(state, 0.0)
    agent.model.load_weights(weights_path)
    print("✓ DRL loaded")
    
    # Initialize trainer
    trainer = GenAITrainer()
    
    print(f"\nCollecting from {num_episodes} episodes (random DC, trained DRL)")
    print(f"Sample interval: every {config.GENAI_SAMPLE_INTERVAL} steps")
    
    start_time = time.time()
    
    for ep in range(num_episodes):
        ep_start = time.time()
        
        state, _ = env.reset(num_dcs=np.random.randint(3, 7))
        done = False
        step_count = 0
        ep_samples = 0
        
        while not done:
            mask = env._get_valid_actions_mask()
            action = agent.get_action(state, epsilon=0.05, valid_actions_mask=mask)
            
            state, reward, done, _, info = env.step(action)
            step_count += 1
            
            # Sample less frequently
            if step_count % config.GENAI_SAMPLE_INTERVAL == 0:
                transitions = env.get_dc_transitions()
                for dc_id, prev_state, curr_state, value in transitions:
                    trainer.collect_transition(
                        env.dcs[dc_id], 
                        env.sfc_manager, 
                        prev_state
                    )
                    ep_samples += 1
        
        # End of episode
        transitions = env.get_dc_transitions()
        for dc_id, prev_state, curr_state, value in transitions:
            trainer.collect_transition(
                env.dcs[dc_id],
                env.sfc_manager,
                prev_state
            )
            ep_samples += 1
        
        ep_time = time.time() - ep_start
        stats = trainer.get_dataset_stats()
        
        # Log after each episode
        print(f"Ep {ep+1:2d}/{num_episodes}: "
              f"samples={ep_samples:3d} | "
              f"total={stats['vae_samples']:5d} | "
              f"time={ep_time:.1f}s")
    
    total_time = time.time() - start_time
    
    final_stats = trainer.get_dataset_stats()
    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETED in {total_time/60:.1f} min")
    print(f"{'='*80}")
    print(f"  VAE samples: {final_stats['vae_samples']}")
    print(f"  Value samples: {final_stats['value_samples']}")
    print(f"  Avg time/episode: {total_time/num_episodes:.1f}s")
    
    return trainer

def train_genai_model(trainer):
    """Train GenAI"""
    print(f"\n{'='*80}")
    print("TRAINING GenAI MODEL")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    trainer.train_vae()
    trainer.train_value_network()
    
    train_time = time.time() - start_time
    
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/genai_model')
    
    print(f"\n{'='*80}")
    print(f"GenAI TRAINING COMPLETED in {train_time/60:.1f} min")
    print(f"{'='*80}")

def main():
    total_start = time.time()
    
    # Collect
    trainer = collect_data()
    
    if trainer.get_dataset_stats()['vae_samples'] < 500:
        print(f"\n⚠ Dataset too small, consider more episodes")
        return
    
    # Train
    train_genai_model(trainer)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"{'='*80}")
    print("\nNext: python scripts.py train --mode genai")

if __name__ == "__main__":
    main()