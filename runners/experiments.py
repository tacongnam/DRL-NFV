import numpy as np
import config
from runners.core import run_single_episode
from runners.visualization import plot_overall_results, plot_scalability_results

def run_experiment_overall(env, agent, episodes=10, file_prefix=""):
    """Overall Performance Analysis (no SFC type breakdown)"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Overall Performance Analysis")
    print(f"Config: {episodes} episodes | Prefix: '{file_prefix}'")
    print(f"{'='*80}")
    
    all_acceptance_ratios = []
    all_avg_delays = []
    all_throughputs = []
    
    for ep in range(episodes):
        print(f"  [Ep {ep+1}/{episodes}] Simulating...", end="\r", flush=True)
        
        env.reset()
        run_single_episode(env, agent, epsilon=0.0, training_mode=False)
        
        stats = env.sfc_manager.get_statistics()
        all_acceptance_ratios.append(stats['acceptance_ratio'])
        all_avg_delays.append(stats['avg_e2e_delay'])
        
        # Calculate throughput
        completed = env.sfc_manager.completed_history
        total_bw = sum(r.bandwidth for r in completed)
        all_throughputs.append(total_bw)
    
    print(f"  [Ep {episodes}/{episodes}] Completed!        ")
    
    avg_ar = np.mean(all_acceptance_ratios)
    avg_delay = np.mean(all_avg_delays)
    avg_throughput = np.mean(all_throughputs)
    
    print(f"\n  Overall Results:")
    print(f"    Acceptance Ratio: {avg_ar:6.2f}%")
    print(f"    Avg E2E Delay:    {avg_delay:6.2f} ms")
    print(f"    Avg Throughput:   {avg_throughput:6.2f} Mbps")
    
    save_path = f'fig/{file_prefix}result_overall.png'
    plot_overall_results(all_acceptance_ratios, all_avg_delays, all_throughputs, save_path=save_path)

def run_experiment_scalability(env, agent, dc_configs, episodes=10, file_prefix=""):
    """Scalability Analysis across different DC counts"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Scalability Analysis")
    print(f"{'='*80}")
    
    results = {
        'dc_counts': [],
        'avg_delays': [],
        'cpu_usages': [],
        'acceptance_ratios': []
    }
    
    for n_dc in dc_configs:
        print(f"\n[Config: {n_dc} DCs]")
        
        episode_ars = []
        episode_delays = []
        episode_cpus = []
        
        for ep in range(episodes):
            print(f"  Running Ep {ep+1}/{episodes}...", end="\r", flush=True)
            
            # Reset with specific DC count (if env supports it)
            env.reset()
            
            state, _ = env._get_obs(), {}
            done = False
            step_cpus = []
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=0.0, valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                # Sample CPU usage
                server_dcs = [dc for dc in env.dcs if dc.is_server]
                if server_dcs:
                    total_cap = sum(config.DC_CPU_CYCLES for dc in server_dcs)
                    used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in server_dcs)
                    step_cpus.append((used_cap / total_cap * 100) if total_cap > 0 else 0)
            
            stats = env.sfc_manager.get_statistics()
            episode_ars.append(stats['acceptance_ratio'])
            
            completed = env.sfc_manager.completed_history
            if completed:
                episode_delays.append(np.mean([r.get_total_e2e_delay() for r in completed]))
            else:
                episode_delays.append(0.0)
            
            if step_cpus:
                episode_cpus.append(np.mean(step_cpus))
        
        avg_ar = np.mean(episode_ars)
        avg_delay = np.mean(episode_delays)
        avg_cpu = np.mean(episode_cpus) if episode_cpus else 0.0
        
        results['dc_counts'].append(n_dc)
        results['acceptance_ratios'].append(avg_ar)
        results['avg_delays'].append(avg_delay)
        results['cpu_usages'].append(avg_cpu)
        
        print(f"  → AR: {avg_ar:.2f}% | E2E: {avg_delay:.2f} ms | CPU: {avg_cpu:.2f}%     ")
    
    save_path = f'fig/{file_prefix}result_scalability.png'
    plot_scalability_results(results, save_path=save_path)