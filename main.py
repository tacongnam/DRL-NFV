#!/usr/bin/env python
"""
Unified CLI for SFC Placement DRL Training and Evaluation

Usage:
    python main.py train dqn --data data/cogent_centers_easy_s1.json
    python main.py train vae --data data/cogent_centers_easy_s1.json
    python main.py eval dqn --data data/cogent_centers_easy_s1.json
    python main.py eval compare --data data/cogent_centers_easy_s1.json
"""

import sys, argparse, os
from envs import PrioritySelector, Observer, VAESelector
from runners import Runner
from agents import VAEAgent
import config, numpy as np

def train_dqn(args):
    """Train standard DQN with Priority selector"""
    Runner.load_from(args.data)
    Runner.train_dqn(dc_selector=PrioritySelector(), save_prefix="", num_updates=args.updates)

def train_vae_dqn(args):
    """Train VAE then DQN with VAE selector"""
    Runner.load_from(args.data)
    
    vae_model = None
    if not args.skip_vae:
        print("\nStep 1/2: Training VAE")
        vae_model = Runner.collect_vae_data(num_episodes=args.vae_episode)
    else:
        print("\nLoading existing VAE model...")
        fake_env = Runner.create_env(dc_selector=PrioritySelector())
        server_dc = fake_env.get_first_dc()
        dc_state_dim = Observer.get_dc_state(server_dc, fake_env.sfc_manager, None).shape[0]

        vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
        vae_model.load_weights("models/vae_model")

        norm_path = "models/vae_model_norm.npz"
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            vae_model.set_normalization_params(float(data['mean']), float(data['std']))
        else:
            print(f"Warning: Normalization parameters not found at {norm_path}. "
                  "VAE predictions might be uncalibrated.")
            
    # Step 2: Train DQN with VAE
    print("\nStep 2/2: Training DQN with VAE selector")
    selector = VAESelector(vae_model)
    Runner.train_dqn(selector, save_prefix="vae_", num_updates=args.updates)

def main(): 
    parser = argparse.ArgumentParser(
        description="DRL-based NFV Placement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['dqn', 'vae'],
                             help='dqn: Priority selector | vae: VAE-assisted')
    train_parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                             help='Training data path')
    train_parser.add_argument('--updates', type=int,
                             help='Training updates (overrides config)')
    train_parser.add_argument('--vae-episodes', type=int,
                             help='VAE collection episodes')
    train_parser.add_argument('--skip-vae', action='store_true',
                             help='Skip VAE training (use existing)')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate models')
    eval_parser.add_argument('mode', choices=['dqn', 'vae'],
                            help='Model to evaluate')
    eval_parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                            help='Test data path')
    eval_parser.add_argument('--episodes', type=int,
                            help='Evaluation episodes')
    eval_parser.add_argument('--model', help='Model path')
    eval_parser.add_argument('--dqn-model', help='DQN model (compare mode)')
    eval_parser.add_argument('--vae-model', help='VAE-DQN model (compare mode)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        if args.mode == 'dqn':
            train_dqn(args)
        else:
            train_vae_dqn(args)

if __name__ == "__main__":
    main()