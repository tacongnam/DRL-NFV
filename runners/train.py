#!/usr/bin/env python
"""Unified training script for DRL and VAE models"""

import argparse
import config
from runners import Runner, load_data
from envs import PrioritySelector, VAESelector, Observer, SFCEnvironment
from agents import VAEAgent
import numpy as np
import os


def train_dqn(args):
    """Train standard DQN with Priority selector"""
    graph, dcs, requests, vnf_specs = load_data(args.data)
    runner = Runner(graph, dcs, requests, vnf_specs)
    selector = PrioritySelector()
    runner.train_dqn(selector, save_prefix="", num_updates=args.updates)


def train_vae_dqn(args):
    """Train VAE then DQN with VAE selector"""
    graph, dcs, requests, vnf_specs = load_data(args.data)
    runner = Runner(graph, dcs, requests, vnf_specs)
    
    # Step 1: Collect VAE data and train
    vae_model = None
    if not args.skip_vae:
        print("\nStep 1/2: Training VAE")
        vae_model = runner.collect_vae_data(num_episodes=args.vae_episodes)
    else:
        print("\nLoading existing VAE model...")
        # --- SỬA LỖI: Lấy dc_state_dim cho khởi tạo VAEAgent ---
        # Cần xác định chiều trạng thái cho một DC duy nhất.
        # Điều này yêu cầu một môi trường hoặc một đối tượng DC giả.
        # Tạo một env tạm thời để lấy state_dim từ Observer.
        temp_env = SFCEnvironment(graph, dcs, requests, PrioritySelector())
        first_server_dc = next((d for d in dcs if d.is_server), None)
        if first_server_dc is None:
            raise ValueError("No server DCs found to determine VAE state dimension.")
        dummy_dc_state = Observer.get_dc_state(first_server_dc, temp_env.sfc_manager, None)
        dc_state_dim = dummy_dc_state.shape[0]
        # --- KẾT THÚC SỬA LỖI ---
        
        vae_model = VAEAgent(dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
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
    runner.train_dqn(selector, save_prefix="vae_", num_updates=args.updates)


def main():
    parser = argparse.ArgumentParser(description="Train SFC Placement Models")
    parser.add_argument('mode', choices=['dqn', 'vae'], 
                       help='Training mode: dqn (Priority) or vae (VAE-assisted)')
    parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                       help='Training data path')
    parser.add_argument('--updates', type=int, default=None,
                       help='Number of training updates (default: config.TRAIN_UPDATES)')
    parser.add_argument('--vae-episodes', type=int, default=None,
                       help='VAE data collection episodes (default: config.GENAI_DATA_EPISODES)')
    parser.add_argument('--skip-vae', action='store_true',
                       help='Skip VAE training, use existing model')
    
    args = parser.parse_args()
    
    if args.mode == 'dqn':
        train_dqn(args)
    else:
        train_vae_dqn(args)


if __name__ == "__main__":
    main()