import os
import glob
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from envs import PrioritySelector, VAESelector, Observer
from agents import DQNAgent, VAEAgent
import config

def calculate_deployment_cost(env):
    """Tính tổng chi phí vận hành (Active Server + Resource Usage)."""
    cost, servers = 0.0, 0
    for dc in env.dcs:
        if dc.is_server and dc.installed_vnfs:
            servers += 1
            # Cost tài nguyên
            res_used = [(dc.initial_resources[k] - getattr(dc, k)) for k in ['cpu', 'ram', 'storage']]
            unit_costs = [dc.cost_c, dc.cost_r, dc.cost_h]
            cost += sum(u * c for u, c in zip(res_used, unit_costs))
            # Cost kích hoạt (Startup)
            cost += (dc.initial_resources['cpu'] * dc.cost_c) * 0.1
    return cost, servers

def run_test_episode(runner, agent_type, model_path, vae_model=None):
    """Chạy 1 episode test cho 1 loại agent."""
    selector = PrioritySelector()
    if agent_type == 'vae':
        selector = VAESelector(vae_model)
    
    env = runner.create_env(selector)
    
    # Load Agent
    state_shapes = [s.shape for s in env.observation_space.spaces]
    agent = DQNAgent(state_shapes, env.action_space.n)
    agent.load(model_path)
    
    state, _ = env.reset()
    done = False
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, 0.0, mask) # Test epsilon = 0
        state, _, done, _, _ = env.step(action)
        
    stats = env.sfc_manager.get_statistics()
    cost, _, _ = calculate_deployment_cost(env)
    return stats['acceptance_ratio'], stats['avg_e2e_delay'], cost

def compare_single_file(runner, data_file, dqn_path, vae_dqn_path, vae_path, episodes):
    runner.load_from(data_file)
    
    # Init VAE Model once
    server_dc = next(d for d in runner.dcs if d.is_server)
    # Lấy state dim thực tế
    dim = Observer.get_dc_state(server_dc, runner.create_env().sfc_manager, None, runner.create_env().topology).shape[0]
    vae = VAEAgent(state_dim=dim, latent_dim=config.GENAI_LATENT_DIM)
    vae.load_weights(vae_path)
    
    # Metrics containers
    res = {'dqn': defaultdict(list), 'vae_dqn': defaultdict(list)}
    
    print(f"Testing {os.path.basename(data_file)} ({episodes} eps)...")
    for _ in range(episodes):
        # Run DQN
        ar, dl, c = run_test_episode(runner, 'dqn', dqn_path)
        res['dqn']['ar'].append(ar); res['dqn']['delay'].append(dl); res['dqn']['cost'].append(c)
        
        # Run VAE-DQN
        ar, dl, c = run_test_episode(runner, 'vae', vae_dqn_path, vae)
        res['vae_dqn']['ar'].append(ar); res['vae_dqn']['delay'].append(dl); res['vae_dqn']['cost'].append(c)

    # Average results
    final = {'file': os.path.basename(data_file)}
    for algo in ['dqn', 'vae_dqn']:
        final[algo] = {k: float(np.mean(v)) for k, v in res[algo].items()}
        
    print(f"  DQN: AR={final['dqn']['ar']:.1f}% | Delay={final['dqn']['delay']:.1f} | Cost={final['dqn']['cost']:.0f}")
    print(f"  VAE: AR={final['vae_dqn']['ar']:.1f}% | Delay={final['vae_dqn']['delay']:.1f} | Cost={final['vae_dqn']['cost']:.0f}")
    return final

def _plot_metric(axes, idx, x, data, title, ylabel):
    """Hàm vẽ biểu đồ dùng chung."""
    width = 0.35
    axes[idx].bar(x - width/2, data['dqn'], width, label='DQN', color='#1f77b4', alpha=0.8)
    axes[idx].bar(x + width/2, data['vae'], width, label='VAE-DQN', color='#ff7f0e', alpha=0.8)
    axes[idx].set_ylabel(ylabel)
    axes[idx].set_title(title)
    axes[idx].set_xticks(x)
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

def compare_all_files(runner, data_folder, dqn_p, vae_dqn_p, vae_p, eps, filter_str='', smart_sample=False):
    files = sorted(glob.glob(os.path.join(data_folder, '*.json')))
    if filter_str: files = [f for f in files if filter_str in os.path.basename(f)]
    
    if smart_sample:
        unique = {}
        for f in files:
            key = "_".join(os.path.basename(f).split('_')[:2]) # Topology_Difficulty
            if key not in unique: unique[key] = f
        files = list(unique.values())

    if not files: return print("No files found.")

    results = []
    for f in files:
        try:
            results.append(compare_single_file(runner, f, dqn_p, vae_dqn_p, vae_p, eps))
        except Exception as e:
            print(f"Err {f}: {e}")

    # Plotting
    labels = [r['file'].replace('_centers', '').replace('.json', '').replace('_s1', '') for r in results]
    metrics = ['ar', 'delay', 'cost']
    titles = ['Acceptance Ratio (%)', 'Avg Delay (ms)', 'Deployment Cost']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    x = np.arange(len(labels))
    
    for i, m in enumerate(metrics):
        data = {'dqn': [r['dqn'][m] for r in results], 'vae': [r['vae_dqn'][m] for r in results]}
        _plot_metric(axes, i, x, data, titles[i], titles[i].split(' ')[0])
        axes[i].set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("fig/comparison_paper_metrics.png", dpi=150)
    print("\nSaved: fig/comparison_paper_metrics.png")
    
    with open('comparison_results.json', 'w') as f:
        json.dump({'files': results}, f, indent=2)