import os
import sys
import numpy as np

# Add parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from environment.genai_env import GenAIEnv
from agent.agent import Agent
from genai.trainer import GenAITrainer

def collect_data(num_episodes=100):
    """
    Thu thập dữ liệu để train GenAI model
    
    Quy trình:
    1. Chạy environment với random DC selection
    2. Sử dụng pre-trained DRL agent để thực hiện actions
    3. Thu thập DC transitions (current_state, next_state, value)
    """
    print("\n" + "="*80)
    print("DATA COLLECTION FOR GenAI TRAINING")
    print("="*80)
    
    # Load pre-trained DRL agent
    agent = Agent()
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if not os.path.exists(weights_path):
        print(f"\n✗ No pre-trained DRL weights found!")
        print("Please train DRL model first: python scripts.py train")
        return
    
    print(f"\nLoading DRL weights: {weights_path}")
    env = GenAIEnv(genai_model=None, data_collection_mode=True)
    state, _ = env.reset()
    agent.get_action(state, 0.0)  # Build model
    agent.model.load_weights(weights_path)
    print("✓ DRL model loaded")
    
    # Initialize trainer
    trainer = GenAITrainer()
    
    # Collect data
    print(f"\nCollecting data from {num_episodes} episodes...")
    print(f"  - Random DC selection for exploration")
    print(f"  - Using pre-trained DRL for actions")
    
    for ep in range(num_episodes):
        state, _ = env.reset(num_dcs=np.random.randint(2, 7))
        done = False
        step_count = 0
        
        while not done:
            # Select action with low epsilon (mostly exploitation)
            mask = env._get_valid_actions_mask()
            action = agent.get_action(state, epsilon=0.1, valid_actions_mask=mask)
            
            # Execute
            state, reward, done, _, info = env.step(action)
            step_count += 1
            
            # Collect transitions periodically
            if step_count % 100 == 0:
                transitions = env.get_dc_transitions()
                
                for dc_id, prev_state, curr_state, value in transitions:
                    trainer.collect_transition(
                        env.dcs[dc_id], 
                        env.sfc_manager, 
                        prev_state
                    )
        
        # End of episode: collect remaining transitions
        transitions = env.get_dc_transitions()
        for dc_id, prev_state, curr_state, value in transitions:
            trainer.collect_transition(
                env.dcs[dc_id],
                env.sfc_manager,
                prev_state
            )
        
        # Progress
        stats = trainer.get_dataset_stats()
        print(f"Ep {ep+1}/{num_episodes}: "
              f"VAE samples={stats['vae_samples']}, "
              f"Value samples={stats['value_samples']}", 
              end="\r", flush=True)
    
    print()  # New line
    
    # Final stats
    final_stats = trainer.get_dataset_stats()
    print(f"\n{'='*80}")
    print(f"DATA COLLECTION COMPLETED")
    print(f"{'='*80}")
    print(f"  VAE dataset: {final_stats['vae_samples']} samples")
    print(f"  Value dataset: {final_stats['value_samples']} samples")
    
    return trainer

def train_genai_model(trainer):
    """Train GenAI model"""
    print(f"\n{'='*80}")
    print("TRAINING GenAI MODEL")
    print(f"{'='*80}")
    
    # Train VAE
    trainer.train_vae(epochs=50, batch_size=64)
    
    # Train Value Network
    trainer.train_value_network(epochs=30, batch_size=64)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/genai_model')
    
    print(f"\n{'='*80}")
    print("GenAI TRAINING COMPLETED")
    print(f"{'='*80}")

def main():
    """Main pipeline"""
    # Step 1: Collect data
    trainer = collect_data(num_episodes=100)
    
    if trainer.get_dataset_stats()['vae_samples'] < 1000:
        print(f"\n⚠ Warning: Dataset too small. Consider collecting more episodes.")
        return
    
    # Step 2: Train GenAI
    train_genai_model(trainer)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Train with GenAI-DRL: python scripts.py train --mode genai")
    print("2. Evaluate: python scripts.py eval --mode genai")

if __name__ == "__main__":
    main()