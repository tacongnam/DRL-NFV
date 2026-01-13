#!/usr/bin/env python
import sys
import os
import argparse
import gc
import tensorflow as tf
import config
from runners import Runner
from envs import PrioritySelector, RandomSelector, VAESelector

# --- Cấu hình GPU cho Kaggle ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s). Memory growth set.", flush=True)
    else:
        print("⚠️ No GPU detected. Training will be slow.", flush=True)
except Exception as e:
    print(f"GPU Config Error: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="DRL-based NFV Placement on Kaggle")
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command (Pipeline mode mặc định cho Kaggle)
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('mode', choices=['pipeline', 'dqn', 'vae'], default='pipeline')
    # Default episodes phù hợp với giới hạn 12h
    train_parser.add_argument('--episodes', type=int, default=config.TRAIN_EPISODES, help='DRL Episodes')
    train_parser.add_argument('--vae-episodes', type=int, default=config.GENAI_DATA_EPISODES, help='VAE Data Episodes')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare DQN vs VAE-DQN')
    compare_parser.add_argument('--data-folder', default='data', help='Folder with test files')
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per file (Test)')
    
    args = parser.parse_args()
    
    if not args.command:
        # Mặc định chạy pipeline nếu không có tham số (tiện cho Kaggle "Run All")
        args.command = 'train'
        args.mode = 'pipeline'
        args.episodes = config.TRAIN_EPISODES
        args.vae_episodes = config.GENAI_DATA_EPISODES

    try:
        if args.command == 'train':
            if args.mode == 'pipeline':
                print(f"\n{'='*80}")
                print(f"🚀 KAGGLE FULL PIPELINE START")
                print(f"   Max Time Estimate: ~6-8 Hours")
                print(f"{'='*80}\n", flush=True)
                
                # --- STEP 1: TRAIN DQN ---
                print(f">>> [1/4] Training DQN ({args.episodes} episodes)...")
                Runner.train_dqn_random(args.episodes, PrioritySelector())
                
                # Clean up memory
                gc.collect()
                tf.keras.backend.clear_session()
                print("   ✔ Memory cleaned after DRL training.\n", flush=True)
                
                # --- STEP 2: COLLECT DATA ---
                print(f">>> [2/4] Collecting VAE Data ({args.vae_episodes} episodes)...")
                Runner.train_vae_random(args.vae_episodes, RandomSelector())
                
                # --- STEP 3: TRAIN VAE IS INCLUDED IN STEP 2 FUNCTION ---
                # Runner.train_vae_random đã bao gồm việc train và save model
                
                gc.collect()
                tf.keras.backend.clear_session()
                print("   ✔ Memory cleaned after VAE training.\n", flush=True)

                # --- STEP 4: AUTO COMPARE ---
                print(f">>> [4/4] Auto-Running Benchmark on 'data/' folder...")
                if os.path.exists('data'):
                    Runner.compare_all_files('data', 'models/best_model', 'models/best_model', 'models/vae_model', 10)
                else:
                    print("   ⚠️ 'data' folder not found. Skipping compare.")

                print(f"\n{'='*80}")
                print(f"✅ PIPELINE FINISHED SUCCESSFULLY")
                print(f"{'='*80}\n")

            elif args.mode == 'dqn':
                Runner.train_dqn_random(args.episodes, PrioritySelector())
            elif args.mode == 'vae':
                Runner.train_vae_random(args.vae_episodes, RandomSelector())
        
        elif args.command == 'compare':
            Runner.compare_all_files(args.data_folder, 'models/best_model', 'models/best_model', 'models/vae_model', args.episodes)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()