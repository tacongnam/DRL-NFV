#!/usr/bin/env python
"""
Unified CLI for SFC Placement DRL Training and Evaluation

Usage:
    python main.py train dqn --data data/cogent_centers_easy_s1.json
    python main.py train vae --data data/cogent_centers_easy_s1.json
    python main.py eval dqn --data data/cogent_centers_easy_s1.json
    python main.py eval compare --data data/cogent_centers_easy_s1.json
"""

import sys
import argparse


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
    eval_parser.add_argument('mode', choices=['dqn', 'vae', 'compare'],
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
        from runners.train import train_dqn, train_vae_dqn
        if args.mode == 'dqn':
            train_dqn(args)
        else:
            train_vae_dqn(args)
    
    elif args.command == 'eval':
        from runners.eval import evaluate_model, compare_models
        if args.mode == 'compare':
            compare_models(args)
        else:
            evaluate_model(args)


if __name__ == "__main__":
    main()