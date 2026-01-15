import os
import glob
import numpy as np
import json
import re
from collections import defaultdict
from envs import PrioritySelector, VAESelector, Observer
from agents import DQNAgent, VAEAgent
import config

def compare_single_file(runner, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes):
    runner.load_from(data_file)
    env = runner.create_env(PrioritySelector())
    
    if hasattr(env.observation_space, 'spaces'):
        state_shapes = [s.shape for s in env.observation_space.spaces]
    else:
        state_shapes = [env.observation_space.shape]
    
    dqn_agent = DQNAgent(state_shapes, env.action_space.n)
    dqn_agent.load(dqn_model_path)
    print(f"DQN model loaded from {dqn_model_path}")
    
    vae_dqn_agent = DQNAgent(state_shapes, env.action_space.n)
    vae_dqn_agent.load(vae_dqn_model_path)
    print(f"VAE-DQN model loaded from {vae_dqn_model_path}")
    
    server_dc = next((d for d in runner.dcs if d.is_server), None)
    dc_state_dim = Observer.get_dc_state(server_dc, env.sfc_manager, None, env.topology).shape[0]
    vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
    vae_model.load_weights(vae_model_path)
    
    norm_path = f"{vae_model_path}_norm.npz"
    if os.path.exists(norm_path):
        data = np.load(norm_path)
        vae_model.set_normalization_params(float(data['mean']), float(data['std']))
    print(f"VAE model loaded from {vae_model_path}\n")
    
    dqn_ars, dqn_delays, dqn_throughputs = [], [], []
    vae_ars, vae_delays, vae_throughputs = [], [], []
    
    print(f"Testing {num_episodes} episodes on {os.path.basename(data_file)}...")
    
    for ep in range(num_episodes):
        # In tiến trình trên cùng 1 dòng (overwrite)
        print(f"\r  Running Episode {ep+1}/{num_episodes}...", end='', flush=True)
        
        # Test DQN
        env_dqn = runner.create_env(PrioritySelector())
        state, _ = env_dqn.reset()
        done = False
        while not done:
            mask = env_dqn._get_valid_actions_mask()
            action = dqn_agent.select_action(state, 0.0, mask)
            state, _, done, _, _ = env_dqn.step(action)
        stats = env_dqn.sfc_manager.get_statistics()
        dqn_ars.append(stats['acceptance_ratio'])
        dqn_delays.append(stats['avg_e2e_delay'])
        dqn_throughputs.append(sum(r.bandwidth for r in env_dqn.sfc_manager.completed_history))
        
        # Test VAE-DQN
        env_vae = runner.create_env(VAESelector(vae_model))
        state, _ = env_vae.reset()
        done = False
        while not done:
            mask = env_vae._get_valid_actions_mask()
            action = vae_dqn_agent.select_action(state, 0.0, mask)
            state, _, done, _, _ = env_vae.step(action)
        stats = env_vae.sfc_manager.get_statistics()
        vae_ars.append(stats['acceptance_ratio'])
        vae_delays.append(stats['avg_e2e_delay'])
        vae_throughputs.append(sum(r.bandwidth for r in env_vae.sfc_manager.completed_history))
    
    # Xóa dòng tiến trình
    print("\r" + " " * 40 + "\r", end='')
    
    # In kết quả tổng hợp của file
    print(f"Results for {os.path.basename(data_file)}:")
    print(f"  DQN     - AR: {np.mean(dqn_ars):.2f}% | Delay: {np.mean(dqn_delays):.2f}ms | TP: {np.mean(dqn_throughputs):.2f}")
    print(f"  VAE-DQN - AR: {np.mean(vae_ars):.2f}% | Delay: {np.mean(vae_delays):.2f}ms | TP: {np.mean(vae_throughputs):.2f}")
    print(f"{'-'*60}")
    
    return {
        'file': os.path.basename(data_file),
        'dqn': {
            'ar': float(np.mean(dqn_ars)), 
            'delay': float(np.mean(dqn_delays)),
            'throughput': float(np.mean(dqn_throughputs))
        },
        'vae_dqn': {
            'ar': float(np.mean(vae_ars)), 
            'delay': float(np.mean(vae_delays)),
            'throughput': float(np.mean(vae_throughputs))
        }
    }

def compare_all_files(runner, data_folder, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes, filter_str='', smart_sample=False):
    print(f"\n{'='*80}")
    print(f"Comparing DQN vs VAE-DQN")
    print(f"Folder: {data_folder}")
    if filter_str: print(f"Filter: '{filter_str}'")
    if smart_sample: print(f"Mode: Smart Sampling (One per Topology+Difficulty)")
    print(f"{'='*80}\n")
    
    all_files = sorted(glob.glob(os.path.join(data_folder, '*.json')))
    target_files = []

    # --- LOGIC LỌC FILE ---
    if smart_sample:
        # Nhóm file theo (Topology, Difficulty) và chỉ lấy 1 file đầu tiên
        seen_categories = set()
        for f in all_files:
            fname = os.path.basename(f)
            
            # Nếu có filter string, kiểm tra trước
            if filter_str and filter_str not in fname:
                continue

            # Parse tên file: cogent_centers_atlanta_easy_s1.json
            # Lấy Topology (cogent) và Difficulty (easy/medium/hard)
            parts = fname.split('_')
            topology = parts[0] # cogent, conus, nsf
            
            difficulty = 'unknown'
            if 'easy' in fname: difficulty = 'easy'
            elif 'medium' in fname: difficulty = 'medium'
            elif 'hard' in fname: difficulty = 'hard'
            
            category = f"{topology}_{difficulty}"
            
            if category not in seen_categories:
                target_files.append(f)
                seen_categories.add(category)
    else:
        # Logic thường: Lấy tất cả hoặc theo filter
        for f in all_files:
            if filter_str in os.path.basename(f):
                target_files.append(f)

    print(f"Selected {len(target_files)} files out of {len(all_files)} for testing.\n")
    
    if len(target_files) == 0:
        print("No files matched. Exiting.")
        return []
    
    all_results = []
    
    for file_idx, data_file in enumerate(target_files):
        print(f"\n{'='*60}")
        print(f"File {file_idx+1}/{len(target_files)}: {os.path.basename(data_file)}")
        print(f"{'='*60}")
        
        try:
            result = compare_single_file(runner, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"OVERALL COMPARISON ({len(all_results)} files)")
    print(f"{'='*80}")
    
    dqn_avg_ar = np.mean([r['dqn']['ar'] for r in all_results])
    dqn_avg_delay = np.mean([r['dqn']['delay'] for r in all_results])
    dqn_avg_tp = np.mean([r['dqn']['throughput'] for r in all_results])
    vae_avg_ar = np.mean([r['vae_dqn']['ar'] for r in all_results])
    vae_avg_delay = np.mean([r['vae_dqn']['delay'] for r in all_results])
    vae_avg_tp = np.mean([r['vae_dqn']['throughput'] for r in all_results])
    
    print(f"DQN Average:")
    print(f"  Acceptance Ratio: {dqn_avg_ar:.2f}%")
    print(f"  E2E Delay: {dqn_avg_delay:.2f}ms")
    print(f"  Throughput: {dqn_avg_tp:.2f}")
    print(f"\nVAE-DQN Average:")
    print(f"  Acceptance Ratio: {vae_avg_ar:.2f}%")
    print(f"  E2E Delay: {vae_avg_delay:.2f}ms")
    print(f"  Throughput: {vae_avg_tp:.2f}")
    print(f"\nImprovement:")
    print(f"  AR: {((vae_avg_ar - dqn_avg_ar) / dqn_avg_ar * 100):.2f}%")
    print(f"  Delay: {((dqn_avg_delay - vae_avg_delay) / dqn_avg_delay * 100):.2f}%")
    print(f"  Throughput: {((vae_avg_tp - dqn_avg_tp) / dqn_avg_tp * 100):.2f}%")
    print(f"{'='*80}\n")
    
    with open('comparison_results.json', 'w') as f:
        json.dump({'files': all_results}, f, indent=2)
        
    _plot_grouped_comparison(all_results)
    
    return all_results

def _parse_filename(filename):
    """Parse <location>_<difficulty>_<scenario>.json"""
    pattern = r'([^_]+)_(easy|medium|hard)_s(\d+)\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return 'unknown', 'unknown', 0

def _plot_grouped_comparison(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        grouped = defaultdict(lambda: {'dqn': {'ar': [], 'delay': [], 'tp': []}, 
                                       'vae': {'ar': [], 'delay': [], 'tp': []}})
        
        for r in results:
            location, difficulty, _ = _parse_filename(r['file'])
            key = f"{location}_{difficulty}"
            grouped[key]['dqn']['ar'].append(r['dqn']['ar'])
            grouped[key]['dqn']['delay'].append(r['dqn']['delay'])
            grouped[key]['dqn']['tp'].append(r['dqn']['throughput'])
            grouped[key]['vae']['ar'].append(r['vae_dqn']['ar'])
            grouped[key]['vae']['delay'].append(r['vae_dqn']['delay'])
            grouped[key]['vae']['tp'].append(r['vae_dqn']['throughput'])
        
        groups = sorted(grouped.keys())
        dqn_ar = [np.mean(grouped[g]['dqn']['ar']) for g in groups]
        vae_ar = [np.mean(grouped[g]['vae']['ar']) for g in groups]
        dqn_delay = [np.mean(grouped[g]['dqn']['delay']) for g in groups]
        vae_delay = [np.mean(grouped[g]['vae']['delay']) for g in groups]
        dqn_tp = [np.mean(grouped[g]['dqn']['tp']) for g in groups]
        vae_tp = [np.mean(grouped[g]['vae']['tp']) for g in groups]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        x = np.arange(len(groups))
        width = 0.35
        
        axes[0].bar(x - width/2, dqn_ar, width, label='DQN', alpha=0.8)
        axes[0].bar(x + width/2, vae_ar, width, label='VAE-DQN', alpha=0.8)
        axes[0].set_ylabel('Acceptance Ratio (%)', fontsize=12)
        axes[0].set_title('Acceptance Ratio by Location & Difficulty', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(groups, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        axes[1].bar(x - width/2, dqn_delay, width, label='DQN', alpha=0.8)
        axes[1].bar(x + width/2, vae_delay, width, label='VAE-DQN', alpha=0.8)
        axes[1].set_ylabel('E2E Delay (ms)', fontsize=12)
        axes[1].set_title('E2E Delay by Location & Difficulty', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(groups, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        axes[2].bar(x - width/2, dqn_tp, width, label='DQN', alpha=0.8)
        axes[2].bar(x + width/2, vae_tp, width, label='VAE-DQN', alpha=0.8)
        axes[2].set_ylabel('Throughput', fontsize=12)
        axes[2].set_title('Throughput by Location & Difficulty', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(groups, rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        os.makedirs("fig", exist_ok=True)
        plt.savefig("fig/comparison_grouped.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("\nGrouped comparison plot saved: fig/comparison_grouped.png")
        
        _plot_by_difficulty(grouped)
        _plot_by_location(grouped)
        
    except Exception as e:
        print(f"Plot error: {e}")

def _plot_by_difficulty(grouped):
    try:
        import matplotlib.pyplot as plt
        
        diff_groups = defaultdict(lambda: {'dqn': {'ar': [], 'delay': [], 'tp': []}, 
                                           'vae': {'ar': [], 'delay': [], 'tp': []}})
        
        for key, data in grouped.items():
            difficulty = key.split('_')[-1]
            diff_groups[difficulty]['dqn']['ar'].extend(data['dqn']['ar'])
            diff_groups[difficulty]['dqn']['delay'].extend(data['dqn']['delay'])
            diff_groups[difficulty]['dqn']['tp'].extend(data['dqn']['tp'])
            diff_groups[difficulty]['vae']['ar'].extend(data['vae']['ar'])
            diff_groups[difficulty]['vae']['delay'].extend(data['vae']['delay'])
            diff_groups[difficulty]['vae']['tp'].extend(data['vae']['tp'])
        
        difficulties = ['easy', 'medium', 'hard']
        dqn_ar = [np.mean(diff_groups[d]['dqn']['ar']) for d in difficulties]
        vae_ar = [np.mean(diff_groups[d]['vae']['ar']) for d in difficulties]
        dqn_delay = [np.mean(diff_groups[d]['dqn']['delay']) for d in difficulties]
        vae_delay = [np.mean(diff_groups[d]['vae']['delay']) for d in difficulties]
        dqn_tp = [np.mean(diff_groups[d]['dqn']['tp']) for d in difficulties]
        vae_tp = [np.mean(diff_groups[d]['vae']['tp']) for d in difficulties]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x = np.arange(len(difficulties))
        width = 0.35
        
        axes[0].bar(x - width/2, dqn_ar, width, label='DQN', alpha=0.8)
        axes[0].bar(x + width/2, vae_ar, width, label='VAE-DQN', alpha=0.8)
        axes[0].set_ylabel('Acceptance Ratio (%)')
        axes[0].set_title('By Difficulty Level', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([d.capitalize() for d in difficulties])
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        axes[1].bar(x - width/2, dqn_delay, width, label='DQN', alpha=0.8)
        axes[1].bar(x + width/2, vae_delay, width, label='VAE-DQN', alpha=0.8)
        axes[1].set_ylabel('E2E Delay (ms)')
        axes[1].set_title('By Difficulty Level', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([d.capitalize() for d in difficulties])
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        axes[2].bar(x - width/2, dqn_tp, width, label='DQN', alpha=0.8)
        axes[2].bar(x + width/2, vae_tp, width, label='VAE-DQN', alpha=0.8)
        axes[2].set_ylabel('Throughput')
        axes[2].set_title('By Difficulty Level', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([d.capitalize() for d in difficulties])
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("fig/comparison_by_difficulty.png", dpi=150)
        plt.close()
        print("Difficulty-based plot saved: fig/comparison_by_difficulty.png")
    except Exception as e:
        print(f"Difficulty plot error: {e}")

def _plot_by_location(grouped):
    try:
        import matplotlib.pyplot as plt
        
        loc_groups = defaultdict(lambda: {'dqn': {'ar': [], 'delay': [], 'tp': []}, 
                                          'vae': {'ar': [], 'delay': [], 'tp': []}})
        
        for key, data in grouped.items():
            location = '_'.join(key.split('_')[:-1])
            loc_groups[location]['dqn']['ar'].extend(data['dqn']['ar'])
            loc_groups[location]['dqn']['delay'].extend(data['dqn']['delay'])
            loc_groups[location]['dqn']['tp'].extend(data['dqn']['tp'])
            loc_groups[location]['vae']['ar'].extend(data['vae']['ar'])
            loc_groups[location]['vae']['delay'].extend(data['vae']['delay'])
            loc_groups[location]['vae']['tp'].extend(data['vae']['tp'])
        
        locations = sorted(loc_groups.keys())
        dqn_ar = [np.mean(loc_groups[loc]['dqn']['ar']) for loc in locations]
        vae_ar = [np.mean(loc_groups[loc]['vae']['ar']) for loc in locations]
        dqn_delay = [np.mean(loc_groups[loc]['dqn']['delay']) for loc in locations]
        vae_delay = [np.mean(loc_groups[loc]['vae']['delay']) for loc in locations]
        dqn_tp = [np.mean(loc_groups[loc]['dqn']['tp']) for loc in locations]
        vae_tp = [np.mean(loc_groups[loc]['vae']['tp']) for loc in locations]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        x = np.arange(len(locations))
        width = 0.35
        
        axes[0].bar(x - width/2, dqn_ar, width, label='DQN', alpha=0.8)
        axes[0].bar(x + width/2, vae_ar, width, label='VAE-DQN', alpha=0.8)
        axes[0].set_ylabel('Acceptance Ratio (%)')
        axes[0].set_title('Acceptance Ratio by Location', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(locations, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        axes[1].bar(x - width/2, dqn_delay, width, label='DQN', alpha=0.8)
        axes[1].bar(x + width/2, vae_delay, width, label='VAE-DQN', alpha=0.8)
        axes[1].set_ylabel('E2E Delay (ms)')
        axes[1].set_title('E2E Delay by Location', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(locations, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        axes[2].bar(x - width/2, dqn_tp, width, label='DQN', alpha=0.8)
        axes[2].bar(x + width/2, vae_tp, width, label='VAE-DQN', alpha=0.8)
        axes[2].set_ylabel('Throughput')
        axes[2].set_title('Throughput by Location', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(locations, rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("fig/comparison_by_location.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Location-based plot saved: fig/comparison_by_location.png")
    except Exception as e:
        print(f"Location plot error: {e}")