import os
import glob
import numpy as np
import json
from envs import PrioritySelector, VAESelector, Observer
from agents import DQNAgent, VAEAgent
import config

def compare_single_file(runner, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes):
    print(f"\n{'='*80}")
    print(f"Comparing DQN vs VAE-DQN on: {os.path.basename(data_file)}")
    print(f"{'='*80}\n")
    
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
    dc_state_dim = Observer.get_dc_state(server_dc, env.sfc_manager, None).shape[0]
    vae_model = VAEAgent(state_dim=dc_state_dim, latent_dim=config.GENAI_LATENT_DIM)
    vae_model.load_weights(vae_model_path)
    
    norm_path = f"{vae_model_path}_norm.npz"
    if os.path.exists(norm_path):
        data = np.load(norm_path)
        vae_model.set_normalization_params(float(data['mean']), float(data['std']))
    print(f"VAE model loaded from {vae_model_path}\n")
    
    dqn_ars, dqn_delays = [], []
    vae_ars, vae_delays = [], []
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}: ", end='', flush=True)
        
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
        
        print(f"DQN AR={dqn_ars[-1]:.1f}% VAE-DQN AR={vae_ars[-1]:.1f}%", flush=True)
    
    print(f"\n{'-'*80}")
    print(f"Results for {os.path.basename(data_file)}:")
    print(f"  DQN     - AR: {np.mean(dqn_ars):.2f}% ± {np.std(dqn_ars):.2f}%, Delay: {np.mean(dqn_delays):.2f}ms ± {np.std(dqn_delays):.2f}ms")
    print(f"  VAE-DQN - AR: {np.mean(vae_ars):.2f}% ± {np.std(vae_ars):.2f}%, Delay: {np.mean(vae_delays):.2f}ms ± {np.std(vae_delays):.2f}ms")
    print(f"{'-'*80}\n")
    
    _plot_comparison(dqn_ars, dqn_delays, vae_ars, vae_delays, os.path.basename(data_file))
    
    return {
        'file': os.path.basename(data_file),
        'dqn': {'ar': float(np.mean(dqn_ars)), 'delay': float(np.mean(dqn_delays))},
        'vae_dqn': {'ar': float(np.mean(vae_ars)), 'delay': float(np.mean(vae_delays))}
    }

def compare_all_files(runner, data_folder, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes):
    print(f"\n{'='*80}")
    print(f"Comparing DQN vs VAE-DQN on all files in: {data_folder}")
    print(f"{'='*80}\n")
    
    data_files = sorted(glob.glob(os.path.join(data_folder, '*.json')))
    print(f"Found {len(data_files)} files\n")
    
    all_results = []
    
    for file_idx, data_file in enumerate(data_files):
        print(f"\n{'='*60}")
        print(f"File {file_idx+1}/{len(data_files)}: {os.path.basename(data_file)}")
        print(f"{'='*60}")
        
        try:
            result = compare_single_file(runner, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"OVERALL COMPARISON ({len(all_results)} files)")
    print(f"{'='*80}")
    
    dqn_avg_ar = np.mean([r['dqn']['ar'] for r in all_results])
    dqn_avg_delay = np.mean([r['dqn']['delay'] for r in all_results])
    vae_avg_ar = np.mean([r['vae_dqn']['ar'] for r in all_results])
    vae_avg_delay = np.mean([r['vae_dqn']['delay'] for r in all_results])
    
    print(f"DQN Average:")
    print(f"  Acceptance Ratio: {dqn_avg_ar:.2f}%")
    print(f"  E2E Delay: {dqn_avg_delay:.2f}ms")
    print(f"\nVAE-DQN Average:")
    print(f"  Acceptance Ratio: {vae_avg_ar:.2f}%")
    print(f"  E2E Delay: {vae_avg_delay:.2f}ms")
    print(f"\nImprovement:")
    print(f"  AR: {((vae_avg_ar - dqn_avg_ar) / dqn_avg_ar * 100):.2f}%")
    print(f"  Delay: {((dqn_avg_delay - vae_avg_delay) / dqn_avg_delay * 100):.2f}%")
    print(f"{'='*80}\n")
    
    with open('comparison_results.json', 'w') as f:
        json.dump({
            'files': all_results,
            'summary': {
                'dqn': {'avg_ar': dqn_avg_ar, 'avg_delay': dqn_avg_delay},
                'vae_dqn': {'avg_ar': vae_avg_ar, 'avg_delay': vae_avg_delay}
            }
        }, f, indent=2)
    
    print("Results saved to comparison_results.json")
    
    _plot_multi_file_comparison(all_results)
    
    return all_results

def _plot_comparison(dqn_ars, dqn_delays, vae_ars, vae_delays, filename):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        episodes = range(1, len(dqn_ars) + 1)
        ax1.plot(episodes, dqn_ars, 'o-', label='DQN', alpha=0.7)
        ax1.plot(episodes, vae_ars, 's-', label='VAE-DQN', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Acceptance Ratio (%)')
        ax1.set_title(f'Acceptance Ratio - {filename}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(episodes, dqn_delays, 'o-', label='DQN', alpha=0.7)
        ax2.plot(episodes, vae_delays, 's-', label='VAE-DQN', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('E2E Delay (ms)')
        ax2.set_title(f'E2E Delay - {filename}')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("fig", exist_ok=True)
        plt.savefig(f"fig/comparison_{filename.replace('.json', '.png')}", dpi=150)
        plt.close()
        print(f"Plot saved: fig/comparison_{filename.replace('.json', '.png')}")
    except Exception as e:
        print(f"Plot error: {e}")

def _plot_multi_file_comparison(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        files = [r['file'] for r in results]
        dqn_ars = [r['dqn']['ar'] for r in results]
        vae_ars = [r['vae_dqn']['ar'] for r in results]
        dqn_delays = [r['dqn']['delay'] for r in results]
        vae_delays = [r['vae_dqn']['delay'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(files))
        width = 0.35
        
        ax1.bar(x - width/2, dqn_ars, width, label='DQN', alpha=0.8)
        ax1.bar(x + width/2, vae_ars, width, label='VAE-DQN', alpha=0.8)
        ax1.set_xlabel('Test Files')
        ax1.set_ylabel('Acceptance Ratio (%)')
        ax1.set_title('Acceptance Ratio Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f[:15] for f in files], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        ax2.bar(x - width/2, dqn_delays, width, label='DQN', alpha=0.8)
        ax2.bar(x + width/2, vae_delays, width, label='VAE-DQN', alpha=0.8)
        ax2.set_xlabel('Test Files')
        ax2.set_ylabel('E2E Delay (ms)')
        ax2.set_title('E2E Delay Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f[:15] for f in files], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        os.makedirs("fig", exist_ok=True)
        plt.savefig("fig/comparison_all_files.png", dpi=150)
        plt.close()
        print("Multi-file comparison plot saved: fig/comparison_all_files.png")
    except Exception as e:
        print(f"Plot error: {e}")