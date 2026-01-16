import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import numpy as np
import json
import matplotlib
matplotlib.use('Agg') # Backend không cần màn hình
import matplotlib.pyplot as plt
from collections import defaultdict
from envs import PrioritySelector, VAESelector, Observer
from agents import DQNAgent, VAEAgent
import config

def calculate_normalized_cost(env):
    """
    Tính Cost chuẩn hóa [0, 1] giống công thức (3), (4) trong bài báo.
    Cost = (Cost thực tế) / (Cost tối đa của hạ tầng)
    """
    current_cost = 0.0
    max_possible_cost = 0.0
    
    # 1. Tính Cost tối đa (Mẫu số): Giả sử toàn bộ server active 100%
    for dc in env.dcs:
        if dc.is_server:
            # Full resource cost
            res_max = [dc.initial_resources['cpu'], dc.initial_resources['ram'], dc.initial_resources['storage']]
            unit_costs = [dc.cost_c, dc.cost_r, dc.cost_h]
            # Cost vận hành tối đa
            op_cost = sum(r * c for r, c in zip(res_max, unit_costs))
            # Cost khởi động (Startup)
            startup_cost = (dc.initial_resources['cpu'] * dc.cost_c) * 0.1
            
            max_possible_cost += (op_cost + startup_cost)

    # 2. Tính Cost hiện tại (Tử số)
    for dc in env.dcs:
        if dc.is_server and len(dc.installed_vnfs) > 0:
            # Resource used
            res_used = [
                (dc.initial_resources['cpu'] - dc.cpu),
                (dc.initial_resources['ram'] - dc.ram),
                (dc.initial_resources['storage'] - dc.storage)
            ]
            unit_costs = [dc.cost_c, dc.cost_r, dc.cost_h]
            
            current_cost += sum(u * c for u, c in zip(res_used, unit_costs))
            current_cost += (dc.initial_resources['cpu'] * dc.cost_c) * 0.1

    if max_possible_cost == 0: return 0.0
    return current_cost / max_possible_cost

def run_test_episode(runner, agent_type, model_path, vae_model=None):
    selector = PrioritySelector()
    if agent_type == 'vae':
        selector = VAESelector(vae_model)
    
    env = runner.create_env(selector)
    
    # Load Agent
    state_shapes = [s.shape for s in env.observation_space.spaces]
    agent = DQNAgent(state_shapes, env.action_space.n)
    
    try:
        agent.load(model_path)
    except:
        pass # Silent fail if model not found (random init)

    state, _ = env.reset()
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, 0.0, mask) # Greedy
        state, _, done, _, _ = env.step(action)
        
    stats = env.sfc_manager.get_statistics()
    norm_cost = calculate_normalized_cost(env)
    
    return stats['acceptance_ratio'], stats['avg_e2e_delay'], norm_cost

def compare_single_file(runner, data_file, dqn_path, vae_dqn_path, vae_path, episodes):
    runner.load_from(data_file)
    
    # Init VAE (Fix lỗi trước đây: phải reset env trước)
    temp_env = runner.create_env()
    temp_env.reset()
    
    try:
        server_dc = temp_env.get_first_dc()
        active_reqs = Observer.get_active_requests(temp_env.sfc_manager)
        g_stats = Observer.precompute_global_stats(temp_env.sfc_manager, active_reqs)
        sample_state = Observer.get_dc_state(server_dc, temp_env.sfc_manager, g_stats, temp_env.topology)
        vae = VAEAgent(state_dim=sample_state.shape[0], latent_dim=config.GENAI_LATENT_DIM)
        vae.load_weights(vae_path)
    except:
        vae = None # Fallback nếu lỗi VAE

    # Metrics containers
    metrics = {
        'dqn': {'ar': [], 'delay': [], 'cost': []},
        'vae': {'ar': [], 'delay': [], 'cost': []}
    }
    
    print(f"Processing {os.path.basename(data_file)}...")
    
    # Chạy N lần để lấy trung bình cho file này (để giảm nhiễu)
    # Bài báo chạy 10 lần/instance. Ở đây ta chạy số episodes quy định.
    for _ in range(episodes):
        # DQN
        ar, dl, c = run_test_episode(runner, 'dqn', dqn_path)
        metrics['dqn']['ar'].append(ar)
        metrics['dqn']['delay'].append(dl)
        metrics['dqn']['cost'].append(c)
        
        # VAE-DQN
        ar, dl, c = run_test_episode(runner, 'vae', vae_dqn_path, vae)
        metrics['vae']['ar'].append(ar)
        metrics['vae']['delay'].append(dl)
        metrics['vae']['cost'].append(c)

    # Trả về trung bình của instance này
    return {
        'file': os.path.basename(data_file),
        'dqn': {k: np.mean(v) for k, v in metrics['dqn'].items()},
        'vae': {k: np.mean(v) for k, v in metrics['vae'].items()}
    }

def plot_paper_style_scatter(results):
    """Vẽ biểu đồ Scatter giống Fig 7 trong bài báo."""
    instances = np.arange(len(results))
    
    # Data Extraction
    dqn_delay = [r['dqn']['delay'] for r in results]
    vae_delay = [r['vae']['delay'] for r in results]
    
    dqn_cost = [r['dqn']['cost'] for r in results]
    vae_cost = [r['vae']['cost'] for r in results]
    
    # Normalizing Delay for Visualization (0.0 - 1.0)
    # Cost đã được normalize trong hàm calculate rồi
    max_delay = max(max(dqn_delay), max(vae_delay)) if dqn_delay else 1.0
    if max_delay == 0: max_delay = 1.0
    
    dqn_delay_norm = [d / max_delay for d in dqn_delay]
    vae_delay_norm = [d / max_delay for d in vae_delay]

    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Subplot 1: Delay Time (Normalized) ---
    ax1.scatter(instances, dqn_delay_norm, color='#1f77b4', label='DQN', alpha=0.8, s=40)
    ax1.scatter(instances, vae_delay_norm, color='#ff7f0e', label='VAE-DQN', alpha=0.8, s=40)
    ax1.set_xlabel('Instances', fontsize=12)
    ax1.set_ylabel('Normalized Delay Time', fontsize=12)
    ax1.set_title('(a) The delay time value', y=-0.15, fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.05, 1.05)

    # --- Subplot 2: Cost of Network Deployment ---
    ax2.scatter(instances, dqn_cost, color='#1f77b4', label='DQN', alpha=0.8, s=40)
    ax2.scatter(instances, vae_cost, color='#ff7f0e', label='VAE-DQN', alpha=0.8, s=40)
    ax2.set_xlabel('Instances', fontsize=12)
    ax2.set_ylabel('Cost of network deployment', fontsize=12)
    ax2.set_title('(b) The cost of network deployment', y=-0.15, fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(-0.05, 1.05) # Cost đã chuẩn hóa 0-1

    plt.tight_layout()
    save_path = "fig/comparison_scatter.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Graph saved to: {save_path}")

def compare_all_files(runner, data_folder, dqn_p, vae_dqn_p, vae_p, eps, filter_str='', smart_sample=False):
    files = sorted(glob.glob(os.path.join(data_folder, '*.json')))
    if filter_str: files = [f for f in files if filter_str in os.path.basename(f)]
    
    if smart_sample:
        unique = {}
        for f in files:
            parts = os.path.basename(f).split('_')
            key = "_".join(parts[:-1]) if len(parts) >= 4 else os.path.basename(f)
            if key not in unique: unique[key] = f
        files = list(unique.values())

    if not files: return print("No files found.")

    results = []
    print(f"Comparing {len(files)} instances...")
    
    for i, f in enumerate(files):
        try:
            r = compare_single_file(runner, f, dqn_p, vae_dqn_p, vae_p, eps)
            results.append(r)
        except Exception as e:
            print(f"Err {f}: {e}")

    if results:
        plot_paper_style_scatter(results)
        
        # Save raw data
        with open('comparison_results.json', 'w') as f:
            json.dump({'files': results}, f, indent=2)