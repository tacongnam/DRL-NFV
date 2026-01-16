#!/usr/bin/env python
import os
import sys
import warnings
import logging

# --- 1. SUPPRESS LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import argparse
import gc
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import config
from runners import Runner
from envs import PrioritySelector

# --- GPU Config ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s).", flush=True)
    else:
        print("⚠️ No GPU detected.", flush=True)
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="DRL-based NFV Placement")
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['pipeline', 'dqn', 'vae'], default='pipeline')
    train_parser.add_argument('--episodes', type=int, default=config.TRAIN_EPISODES, help='DRL Episodes')
    
    # --- UPDATE: Thêm tham số cho VAE ---
    train_parser.add_argument('--vae-episodes', type=int, default=config.GENAI_DATA_EPISODES, help='Episodes to collect VAE data')
    train_parser.add_argument('--vae-epochs', type=int, default=config.GENAI_VAE_EPOCHS, help='VAE Training Epochs')
    
    compare_parser = subparsers.add_parser('compare', help='Compare DQN vs VAE-DQN')
    compare_parser.add_argument('--data-folder', default='data', help='Folder with test files')
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per file (Test)')
    compare_parser.add_argument('--filter', type=str, default='', help='Filter files')
    compare_parser.add_argument('--smart-sample', action='store_true', help='Smart sampling')
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'train'
        args.mode = 'pipeline'
        args.episodes = config.TRAIN_EPISODES
        args.vae_episodes = config.GENAI_DATA_EPISODES
        args.vae_epochs = config.GENAI_VAE_EPOCHS

    try:
        if args.command == 'train':
            if args.mode == 'pipeline':
                print(f"\n{'='*80}\n🚀 FULL PIPELINE START\n{'='*80}\n", flush=True)
                
                # Step 1: DQN
                print(f">>> [1/3] Training DQN ({args.episodes} eps)...")
                Runner.train_dqn_random(args.episodes, PrioritySelector())
                
                gc.collect()
                tf.keras.backend.clear_session()
                
                # Step 2: VAE (Truyền vae_epochs vào)
                print(f">>> [2/3] Collecting & Training VAE ({args.vae_episodes} eps, {args.vae_epochs} epochs)...")
                Runner.train_vae_random(args.vae_episodes, PrioritySelector(), vae_epochs=args.vae_epochs)
                
                gc.collect()
                tf.keras.backend.clear_session()

                # Step 3: Compare
                print(f">>> [3/3] Auto-Running Benchmark...")
                if os.path.exists('data'):
                    Runner.compare_all_files('data', 'models/best_model', 'models/best_model', 'models/vae_model', 10, filter_str='', smart_sample=True)
                
                print(f"\n✅ PIPELINE FINISHED")

            elif args.mode == 'dqn':
                Runner.train_dqn_random(args.episodes, PrioritySelector())
            elif args.mode == 'vae':
                # Truyền vae_epochs vào đây
                Runner.train_vae_random(args.vae_episodes, PrioritySelector(), vae_epochs=args.vae_epochs)
        
        elif args.command == 'compare':
            Runner.compare_all_files(args.data_folder, 'models/best_model', 'models/best_model', 'models/vae_model', args.episodes, filter_str=args.filter, smart_sample=args.smart_sample)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()