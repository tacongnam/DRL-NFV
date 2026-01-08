#!/usr/bin/env python
import sys
import os
import argparse
import config
from runners import Runner
from agents import DQNAgent, VAEAgent
from envs import PrioritySelector, RandomSelector, VAESelector, Observer

def main():
    parser = argparse.ArgumentParser(description="DRL-based NFV Placement")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['dqn', 'vae', 'random', 'pipeline'])
    train_parser.add_argument('--data', default='data/test.json', help='Data file for dqn/vae mode')
    train_parser.add_argument('--updates', type=int, default=40, help='Updates for dqn mode')
    train_parser.add_argument('--episodes', type=int, default=500, help='Episodes for random mode')
    train_parser.add_argument('--vae-episodes', type=int, default=200, help='Episodes for vae mode')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare DQN vs VAE-DQN')
    compare_parser.add_argument('--data', help='Single file (optional, default: all files in data/)')
    compare_parser.add_argument('--data-folder', default='data', help='Folder with test files')
    compare_parser.add_argument('--episodes', type=int, default=5, help='Episodes per file')
    compare_parser.add_argument('--dqn-model', default='models/best_model')
    compare_parser.add_argument('--vae-dqn-model', default='models/best_model')
    compare_parser.add_argument('--vae-model', default='models/vae_model')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            if args.mode == 'random':
                print(f"Training DQN with {args.episodes} random episodes", flush=True)
                Runner.train_dqn_random(args.episodes, PrioritySelector())
            
            elif args.mode == 'dqn':
                print(f"Training DQN on {args.data} for {args.updates} updates", flush=True)
                Runner.train_dqn_file(args.data, args.updates, PrioritySelector())
            
            elif args.mode == 'vae':
                print(f"Training VAE with {args.vae_episodes} random episodes", flush=True)
                Runner.train_vae_random(args.vae_episodes, RandomSelector())
            
            elif args.mode == 'pipeline':
                print(f"\n{'='*80}")
                print(f"FULL TRAINING PIPELINE")
                print(f"{'='*80}")
                print(f"Step 1: Train DQN ({args.episodes} episodes)")
                print(f"Step 2: Collect VAE data ({args.vae_episodes} episodes)")
                print(f"Step 3: Train VAE models")
                print(f"{'='*80}\n")
                
                print(f"\n>>> STEP 1: Training DQN with random scenarios...")
                Runner.train_dqn_random(args.episodes, PrioritySelector())
                
                print(f"\n>>> STEP 2: Collecting VAE data using trained DQN...")
                Runner.train_vae_random(args.vae_episodes, RandomSelector())
                
                print(f"\n{'='*80}")
                print(f"PIPELINE COMPLETE!")
                print(f"  DQN model: models/best_model")
                print(f"  VAE model: models/vae_model")
                print(f"{'='*80}\n")
        
        elif args.command == 'compare':
            if args.data:
                print(f"Comparing on single file: {args.data}", flush=True)
                Runner.compare_single_file(args.data, args.dqn_model, args.vae_dqn_model, 
                                          args.vae_model, args.episodes)
            else:
                print(f"Comparing on all files in {args.data_folder}", flush=True)
                Runner.compare_all_files(args.data_folder, args.dqn_model, args.vae_dqn_model, 
                                        args.vae_model, args.episodes)
        
        print("\nCompleted successfully!", flush=True)
    
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()