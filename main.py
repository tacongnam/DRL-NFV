#!/usr/bin/env python
import os
import sys
import warnings
import logging

# --- 1. SUPPRESS TENSORFLOW & CUDA LOGS ---
# Phải đặt biến môi trường TRƯỚC KHI import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL only (ẩn Info, Warning, Error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt OneDNN logs nếu có

# --- 2. SUPPRESS GYM & DEPRECATION WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Lọc cụ thể thông báo về Gym unmaintained
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# --- 3. IMPORTS ---
import argparse
import gc
import tensorflow as tf

# Tắt logger của TF python
tf.get_logger().setLevel(logging.ERROR)
# Tắt absl logging (thư viện Google log mà TF dùng)
logging.getLogger('absl').setLevel(logging.ERROR)

import config
from runners import Runner
from envs import PrioritySelector, RandomSelector, VAESelector

# --- Cấu hình GPU cho Kaggle ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth error: {e}")
        print(f"✅ GPU Detected: {len(gpus)} device(s).", flush=True)
    else:
        print("⚠️ No GPU detected. Training will be slow.", flush=True)
except Exception as e:
    print(f"GPU Config Error: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="DRL-based NFV Placement on Kaggle")
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['pipeline', 'dqn', 'vae'], default='pipeline')
    train_parser.add_argument('--episodes', type=int, default=config.TRAIN_EPISODES, help='DRL Episodes')
    train_parser.add_argument('--vae-episodes', type=int, default=config.GENAI_DATA_EPISODES, help='VAE Data Episodes')
    
    compare_parser = subparsers.add_parser('compare', help='Compare DQN vs VAE-DQN')
    compare_parser.add_argument('--data-folder', default='data', help='Folder with test files')
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per file (Test)')
    compare_parser.add_argument('--filter', type=str, default='', help='Filter files by name (e.g., "hard", "cogent")')
    compare_parser.add_argument('--smart-sample', action='store_true', help='Only run 1 file per Topology+Difficulty combination')
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'train'
        args.mode = 'pipeline'
        args.episodes = config.TRAIN_EPISODES
        args.vae_episodes = config.GENAI_DATA_EPISODES

    try:
        if args.command == 'train':
            if args.mode == 'pipeline':
                print(f"\n{'='*80}")
                print(f"🚀 KAGGLE FULL PIPELINE START")
                print(f"{'='*80}\n", flush=True)
                
                # --- STEP 1: TRAIN DQN ---
                print(f">>> [1/4] Training DQN ({args.episodes} episodes)...")
                Runner.train_dqn_random(args.episodes, PrioritySelector())
                
                gc.collect()
                tf.keras.backend.clear_session()
                print("   ✔ Memory cleaned.\n", flush=True)
                
                # --- STEP 2: COLLECT & TRAIN VAE ---
                print(f">>> [2/4] Collecting & Training VAE ({args.vae_episodes} episodes)...")
                Runner.train_vae_random(args.vae_episodes, PrioritySelector())
                
                gc.collect()
                tf.keras.backend.clear_session()
                print("   ✔ Memory cleaned.\n", flush=True)

                # --- STEP 4: AUTO COMPARE ---
                print(f">>> [4/4] Auto-Running Benchmark...")
                if os.path.exists('data'):
                    Runner.compare_all_files(
                        'data', 'models/best_model', 'models/best_model', 'models/vae_model', 
                        10, filter_str='', smart_sample=True
                    )
                else:
                    print("   ⚠️ 'data' folder not found. Skipping compare.")

                print(f"\n{'='*80}")
                print(f"✅ PIPELINE FINISHED SUCCESSFULLY")
                print(f"{'='*80}\n")

            elif args.mode == 'dqn':
                Runner.train_dqn_random(args.episodes, PrioritySelector())
            elif args.mode == 'vae':
                Runner.train_vae_random(args.vae_episodes, PrioritySelector())
        
        elif args.command == 'compare':
            Runner.compare_all_files(
                args.data_folder, 
                'models/best_model', 
                'models/best_model', 
                'models/vae_model', 
                args.episodes,
                filter_str=args.filter,
                smart_sample=args.smart_sample
            )
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()