import os
# --- Tắt GPU để tăng tốc độ Test từng mẫu (Batch size=1 chạy CPU nhanh hơn) ---
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
import csv

def calculate_normalized_cost(env):
    """Tính Cost chuẩn hóa [0, 1]."""
    current_cost = 0.0
    max_possible_cost = 0.0
    
    # 1. Cost tối đa (Mẫu số)
    for dc in env.dcs:
        if dc.is_server:
            res_max = [dc.initial_resources['cpu'], dc.initial_resources['ram'], dc.initial_resources['storage']]
            unit_costs = [dc.cost_c, dc.cost_r, dc.cost_h]
            op_cost = sum(r * c for r, c in zip(res_max, unit_costs))
            startup_cost = (dc.initial_resources['cpu'] * dc.cost_c) * 0.1
            max_possible_cost += (op_cost + startup_cost)

    # 2. Cost hiện tại (Tử số)
    for dc in env.dcs:
        if dc.is_server and len(dc.installed_vnfs) > 0:
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
    state_shapes = [s.shape for s in env.observation_space.spaces]
    agent = DQNAgent(state_shapes, env.action_space.n)
    
    try:
        agent.load(model_path)
    except:
        pass 

    state, _ = env.reset()
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, 0.0, mask) # Greedy
        state, _, done, _, _ = env.step(action)
        
    stats = env.sfc_manager.get_statistics()
    norm_cost = calculate_normalized_cost(env)
    
    # Tính Throughput: Tổng băng thông các request đã hoàn thành
    total_throughput = sum(req.bandwidth for req in env.sfc_manager.completed_history)
    
    return stats['acceptance_ratio'], stats['avg_e2e_delay'], norm_cost, total_throughput

def compare_single_file(runner, data_file, dqn_path, vae_dqn_path, vae_path, episodes, log_writer=None):
    runner.load_from(data_file)
    
    # Init VAE
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
        vae = None

    metrics = {
        'dqn': {'ar': [], 'delay': [], 'cost': [], 'tp': []},
        'vae': {'ar': [], 'delay': [], 'cost': [], 'tp': []}
    }
    
    file_name = os.path.basename(data_file)
    print(f"Testing {file_name}...", end=" ", flush=True)
    
    for _ in range(episodes):
        # Run DQN
        ar, dl, c, tp = run_test_episode(runner, 'dqn', dqn_path)
        metrics['dqn']['ar'].append(ar); metrics['dqn']['delay'].append(dl)
        metrics['dqn']['cost'].append(c); metrics['dqn']['tp'].append(tp)
        
        # Run VAE-DQN
        ar, dl, c, tp = run_test_episode(runner, 'vae', vae_dqn_path, vae)
        metrics['vae']['ar'].append(ar); metrics['vae']['delay'].append(dl)
        metrics['vae']['cost'].append(c); metrics['vae']['tp'].append(tp)

    # Average
    avg_dqn = {k: np.mean(v) for k, v in metrics['dqn'].items()}
    avg_vae = {k: np.mean(v) for k, v in metrics['vae'].items()}
    
    # Print Log
    print(f"\n  > DQN: AR={avg_dqn['ar']:.1f}% | Delay={avg_dqn['delay']:.2f} | TP={avg_dqn['tp']:.0f}")
    print(f"  > VAE: AR={avg_vae['ar']:.1f}% | Delay={avg_vae['delay']:.2f} | TP={avg_vae['tp']:.0f}")
    
    # Save to CSV Log
    if log_writer:
        log_writer.writerow([
            file_name, 
            f"{avg_dqn['ar']:.2f}", f"{avg_vae['ar']:.2f}",
            f"{avg_dqn['delay']:.4f}", f"{avg_vae['delay']:.4f}",
            f"{avg_dqn['tp']:.2f}", f"{avg_vae['tp']:.2f}",
            f"{avg_dqn['cost']:.4f}", f"{avg_vae['cost']:.4f}"
        ])

    return {
        'file': file_name,
        'dqn': avg_dqn,
        'vae': avg_vae
    }

def plot_performance_comparison(results):
    """Vẽ 3 biểu đồ cột: AR, Delay, Throughput."""
    labels = []
    # Rút gọn tên file để hiển thị
    for r in results:
        name = r['file'].replace('.json', '').replace('cogent', 'CG').replace('conus', 'CN').replace('nsf', 'NSF')
        name = name.replace('_centers', '-C').replace('_uniform', '-U').replace('_rural', '-R').replace('_urban', '-Ub')
        name = name.replace('_hard_s1', '-H').replace('_easy_s1', '-E').replace('_normal_s1', '-N')
        labels.append(name)
        
    x = np.arange(len(labels))
    width = 0.35

    # Data
    ar_dqn = [r['dqn']['ar'] for r in results]
    ar_vae = [r['vae']['ar'] for r in results]
    
    dl_dqn = [r['dqn']['delay'] for r in results]
    dl_vae = [r['vae']['delay'] for r in results]
    
    tp_dqn = [r['dqn']['tp'] for r in results]
    tp_vae = [r['vae']['tp'] for r in results]

    # --- PLOT ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # 1. Acceptance Ratio
    axes[0].bar(x - width/2, ar_dqn, width, label='DQN', color='#d62728', alpha=0.8)
    axes[0].bar(x + width/2, ar_vae, width, label='VAE-DQN', color='#2ca02c', alpha=0.8)
    axes[0].set_ylabel('Acceptance Ratio (%)')
    axes[0].set_title('Acceptance Ratio Comparison', fontweight='bold')
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # 2. Delay
    axes[1].bar(x - width/2, dl_dqn, width, label='DQN', color='#1f77b4', alpha=0.8)
    axes[1].bar(x + width/2, dl_vae, width, label='VAE-DQN', color='#ff7f0e', alpha=0.8)
    axes[1].set_ylabel('Avg End-to-End Delay (ms)')
    axes[1].set_title('Average Delay Comparison (Lower is Better)', fontweight='bold')
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    # 3. Throughput
    axes[2].bar(x - width/2, tp_dqn, width, label='DQN', color='#9467bd', alpha=0.8)
    axes[2].bar(x + width/2, tp_vae, width, label='VAE-DQN', color='#8c564b', alpha=0.8)
    axes[2].set_ylabel('Total Throughput')
    axes[2].set_title('Total Throughput Comparison (Higher is Better)', fontweight='bold')
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("fig/performance_comparison.png", dpi=200)
    print("✅ Saved detailed performance plots to: fig/performance_comparison.png")

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

    # Mở file CSV log
    log_file = open("results_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Filename", "AR_DQN", "AR_VAE", "Delay_DQN", "Delay_VAE", "TP_DQN", "TP_VAE", "Cost_DQN", "Cost_VAE"])

    results = []
    print(f"Starting comparison on {len(files)} files...")
    
    for f in files:
        try:
            r = compare_single_file(runner, f, dqn_p, vae_dqn_p, vae_p, eps, log_writer)
            results.append(r)
        except Exception as e:
            print(f"\n❌ Err {os.path.basename(f)}: {e}")
    
    log_file.close()
    print(f"\n✅ Log saved to results_log.csv")

    if results:
        # Plot chi tiết (Bar charts)
        plot_performance_comparison(results)
        
        # Save JSON
        with open('comparison_results.json', 'w') as f:
            json.dump({'files': results}, f, indent=2)