import numpy as np
import time
import config
# Import từ core và visualization
from runners.core import run_single_episode
from runners.visualization import plot_exp1_results, plot_exp2_results

def run_experiment_performance(env, agent, episodes=10):
    """Chạy Experiment 1: Performance Analysis"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Performance Analysis per SFC Type")
    print(f"{'='*80}")
    
    total_completed = []
    total_dropped = []
    
    for ep in range(episodes):
        print(f"[Ep {ep+1}/{episodes}]", end="\r", flush=True)
        env.reset(num_dcs=4)
        run_single_episode(env, agent, epsilon=config.TEST_EPSILON, training_mode=False)
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
    
    print("\nProcessing results...")
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    for sfc_type in sfc_types:
        completed = [r for r in total_completed if r.type == sfc_type]
        dropped = [r for r in total_dropped if r.type == sfc_type]
        total = len(completed) + len(dropped)
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.get_total_e2e_delay() for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        print(f"  {sfc_type:15s}: AR={ar:6.2f}%  |  E2E Delay={avg_delay:6.2f} ms")
    
    plot_exp1_results(sfc_types, acc_ratios, e2e_delays)

def run_experiment_scalability(env, agent, episodes=10):
    """Chạy Experiment 2: Scalability"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: Reconfigurability & Scalability")
    print(f"{'='*80}")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        print(f"\n[Config: {n_dc} DCs]")
        current_completed = []
        cpu_usages = []
        
        for ep in range(episodes):
            print(f"  Ep {ep+1}/{episodes}", end="\r", flush=True)
            env.reset(num_dcs=n_dc)
            state, _ = env._get_obs(), {}
            done = False
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=config.TEST_EPSILON, valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                total_cap = n_dc * config.DC_CPU_CYCLES
                used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in env.dcs)
                cpu_usages.append((used_cap / total_cap * 100) if total_cap > 0 else 0)
            
            current_completed.extend(env.sfc_manager.completed_history)
        
        avg_delay = np.mean([r.get_total_e2e_delay() for r in current_completed]) if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        print(f"  → Avg E2E: {avg_delay:.2f} ms | Avg CPU: {avg_cpu:.2f}%")
    
    plot_exp2_results(dc_counts, exp2_delays, exp2_resources)