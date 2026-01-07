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
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['dqn', 'vae', 'random'])
    train_parser.add_argument('--data', default='data/test.json', help='Data file for dqn/vae mode')
    train_parser.add_argument('--updates', type=int, default=40, help='Updates for dqn mode')
    train_parser.add_argument('--episodes', type=int, default=500, help='Episodes for random/vae mode')
    
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
                print(f"Training VAE with {args.episodes} random episodes", flush=True)
                Runner.train_vae_random(args.episodes, RandomSelector())
        
        elif args.command == 'compare':
            if args.data:
                print(f"Comparing on single file: {args.data}", flush=True)
                Runner.compare_single_file(args.data, args.dqn_model, args.vae_dqn_model, args.vae_model, args.episodes)
            else:
                print(f"Comparing on all files in {args.data_folder}", flush=True)
                Runner.compare_all_files(args.data_folder, args.dqn_model, args.vae_dqn_model, args.vae_model, args.episodes)
        
        print("\nCompleted successfully!", flush=True)
    
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()