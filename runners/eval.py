#!/usr/bin/env python
"""Unified evaluation script for DRL and VAE models""" # Đổi mô tả

import argparse
import config
from runners import Runner, load_data # Sửa import
from envs import PrioritySelector, VAESelector, Observer, SFCEnvironment
from agents import DQNAgent, VAEAgent # Thêm import này
import numpy as np
import os


def evaluate_dqn(args): # Đổi tên hàm
    """Evaluate standard DQN with Priority selector"""
    graph, dcs, requests, vnf_specs = load_data(args.data)
    runner = Runner(graph, dcs, requests, vnf_specs)
    
    # --- SỬA LỖI: Trích xuất đúng shape cho DQN đa đầu vào ---
    # Cần tạo một env giả để lấy shape của không gian quan sát
    temp_env = SFCEnvironment(graph, dcs, requests, PrioritySelector())
    if hasattr(temp_env.observation_space, 'spaces'):
        dqn_state_shapes = [s.shape for s in temp_env.observation_space.spaces]
    else:
        dqn_state_shapes = [temp_env.observation_space.shape]
    # --- KẾT THÚC SỬA LỖI ---
    
    agent = DQNAgent(dqn_state_shapes, temp_env.action_space.n)
    agent.load(f"models/{args.model_path}") # Tải mô hình DQN đã huấn luyện
    
    selector = PrioritySelector()
    runner.evaluate(agent, selector, num_episodes=args.episodes, prefix="dqn_priority_")


def evaluate_vae_dqn(args): # Đổi tên hàm
    """Evaluate VAE-assisted DQN with VAE selector"""
    graph, dcs, requests, vnf_specs = load_data(args.data)
    runner = Runner(graph, dcs, requests, vnf_specs)
    
    # Step 1: Tải mô hình VAE
    print("\nLoading VAE model...")
    # --- SỬA LỖI: Lấy dc_state_dim cho khởi tạo VAEAgent ---
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
    
    # Step 2: Tải DQN với VAE selector
    print("\nLoading DQN model with VAE selector")
    
    # --- SỬA LỖI: Trích xuất đúng shape cho DQN đa đầu vào ---
    # Cần tạo một env giả để lấy shape của không gian quan sát
    if hasattr(temp_env.observation_space, 'spaces'):
        dqn_state_shapes = [s.shape for s in temp_env.observation_space.spaces]
    else:
        dqn_state_shapes = [temp_env.observation_space.shape]
    # --- KẾT THÚC SỬA LỖI ---
    
    dqn_agent = DQNAgent(dqn_state_shapes, temp_env.action_space.n)
    dqn_agent.load(f"models/{args.model_path}") # Tải mô hình DQN đã huấn luyện
    
    selector = VAESelector(vae_model)
    runner.evaluate(dqn_agent, selector, num_episodes=args.episodes, prefix="vae_dqn_")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFC Placement Models")
    parser.add_argument('mode', choices=['dqn', 'vae'], 
                       help='Evaluation mode: dqn (Priority) or vae (VAE-assisted)')
    parser.add_argument('--data', default='data/cogent_centers_easy_s1.json',
                       help='Evaluation data path')
    parser.add_argument('--model-path', required=True,
                       help='Path to the trained DQN model weights (e.g., "best_model")')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of evaluation episodes (default: config.TEST_EPISODES)')
    
    args = parser.parse_args()
    
    if args.mode == 'dqn':
        evaluate_dqn(args)
    else:
        evaluate_vae_dqn(args)


if __name__ == "__main__":
    main()