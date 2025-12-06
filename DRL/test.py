import numpy as np
import matplotlib.pyplot as plt
import os
import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN

# DEBUG MODE: Set to True to see detailed action logs
DEBUG_MODE = True
DEBUG_TIMING = False  # Set to True to see detailed timing information

def plot_exp1(types, ars, delays):
    """Vẽ biểu đồ cho Thực nghiệm 1 (Fig 2)"""
    if not types:
        print("No data to plot for Exp 1.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(types))
    width = 0.35
    
    # Bar Chart: Acceptance Ratio
    ax1.bar(x, ars, width, label='Acceptance Ratio (%)', color='b', alpha=0.6)
    ax1.set_xlabel('SFC Types')
    ax1.set_ylabel('Acceptance Ratio (%)', color='b')
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Line Chart: E2E Delay
    ax2 = ax1.twinx()
    ax2.plot(x, delays, color='r', marker='o', linewidth=2, label='E2E Delay (ms)')
    ax2.set_ylabel('Avg E2E Delay (ms)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    max_delay = max(delays) if delays else 100
    ax2.set_ylim(0, max_delay * 1.2)
    
    plt.title('Experiment 1: Performance per SFC Type (Fixed 4 DCs)')
    fig.tight_layout()
    plt.savefig('fig/result_exp1_fig2.png')
    print("Saved result_exp1_fig2.png")

def plot_exp2(dc_counts, delays, resources):
    """Vẽ biểu đồ cho Thực nghiệm 2 (Fig 3)"""
    if not dc_counts:
        print("No data to plot for Exp 2.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graph 1: Delay vs DC Count
    ax1.plot(dc_counts, delays, 'g-o', linewidth=2)
    ax1.set_title('E2E Delay vs Number of DCs')
    ax1.set_xlabel('Number of DCs')
    ax1.set_ylabel('Avg E2E Delay (ms)')
    ax1.set_xticks(dc_counts)
    ax1.grid(True)
    
    # Graph 2: Resource vs DC Count
    ax2.bar(dc_counts, resources, color='orange', alpha=0.7, width=1.0)
    ax2.set_title('Avg Resource Consumption vs Number of DCs')
    ax2.set_xlabel('Number of DCs')
    ax2.set_ylabel('Avg CPU Usage (%)')
    ax2.set_xticks(dc_counts)
    ax2.grid(True, axis='y')
    
    plt.suptitle('Experiment 2: Reconfigurability & Robustness')
    plt.tight_layout()
    plt.savefig('fig/result_exp2_fig3.png')
    print("Saved result_exp2_fig3.png")

def run_experiment_1(env, agent):
    print("\n>>> RUNNING EXPERIMENT 1: Performance Analysis per SFC Type (4 DCs)")
    
    total_completed_history = []
    total_dropped_history = []
    
    # DEBUG: Track action distribution
    action_counts = {}
    
    for ep in range(config.TEST_EPISODES):
        state, _ = env.reset(num_dcs=4)
        action_mask = env._get_valid_actions_mask()
        done = False
        step_count = 0
        total_reward = 0
        
        if DEBUG_MODE and ep == 0:
            print(f"\n  [DEBUG] Episode {ep+1} starting...")
            print(f"  [DEBUG] Initial requests: {len(env.sfc_manager.active_requests)}")
            if DEBUG_TIMING:
                timing = env.get_timing_info()
                print(f"  [DEBUG] Timing: {timing['action_inference_time_ms']:.4f}ms per action")
                print(f"  [DEBUG] {timing['actions_per_timestep']} actions = 1 timestep = 1ms physical time")
        
        while not done:
            action = agent.get_action(state, epsilon=config.TEST_EPSILON, valid_actions_mask=action_mask)
            
            # DEBUG: Track actions
            if DEBUG_MODE and ep == 0 and step_count < 50:  # First 50 steps only
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
                
                if step_count % 10 == 0:
                    action_type = "WAIT" if action == 0 else \
                                 f"UNINSTALL-{config.VNF_TYPES[action-1]}" if action <= config.NUM_VNF_TYPES else \
                                 f"ALLOC-{config.VNF_TYPES[action - config.NUM_VNF_TYPES - 1]}"
                    
                    if DEBUG_TIMING:
                        timing = env.get_timing_info()
                        print(f"    Step {step_count}: Action={action_type}, "
                              f"SimTime={timing['sim_time_ms']}ms, "
                              f"ActionInTimestep={timing['action_counter']}/{timing['actions_per_timestep']}, "
                              f"Pending={len(env.sfc_manager.active_requests)}")
                    else:
                        print(f"    Step {step_count}: Action={action_type}, Pending={len(env.sfc_manager.active_requests)}")
            
            state, reward, done, _, info = env.step(action)
            action_mask = info.get('action_mask', None)
            total_reward += reward
            step_count += 1
        
        total_completed_history.extend(env.sfc_manager.completed_history)
        total_dropped_history.extend(env.sfc_manager.dropped_history)
        
        print(f"\r   > Ep {ep+1}/{config.TEST_EPISODES} completed. Reward: {total_reward:.1f}. "
              f"Requests (Done/Drop): {len(env.sfc_manager.completed_history)}/{len(env.sfc_manager.dropped_history)}", end="")
    
    print("")
    
    if DEBUG_MODE:
        print("\n  [DEBUG] Action Distribution (First Episode):")
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for action, count in sorted_actions:
            action_type = "WAIT" if action == 0 else \
                         f"UNINSTALL-{config.VNF_TYPES[action-1]}" if action <= config.NUM_VNF_TYPES else \
                         f"ALLOC-{config.VNF_TYPES[action - config.NUM_VNF_TYPES - 1]}"
            print(f"    {action_type}: {count} times")

    # Tổng hợp số liệu
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    print("-" * 75)
    print(f"{'SFC Type':<15} | {'Acc Ratio (%)':<15} | {'Avg Delay (ms)':<15} | {'Count (Done/All)'}")
    print("-" * 75)
    
    for s_type in sfc_types:
        completed = [r for r in total_completed_history if r.type == s_type]
        dropped = [r for r in total_dropped_history if r.type == s_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.elapsed_time for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        print(f"{s_type:<15} | {ar:<15.2f} | {avg_delay:<15.2f} | {len(completed)}/{total}")
    
    print("-" * 75)
    overall_ar = (len(total_completed_history) / (len(total_completed_history) + len(total_dropped_history)) * 100) if (len(total_completed_history) + len(total_dropped_history)) > 0 else 0.0
    print(f"{'OVERALL':<15} | {overall_ar:<15.2f} | {'N/A':<15} | {len(total_completed_history)}/{len(total_completed_history) + len(total_dropped_history)}")
    
    plot_exp1(sfc_types, acc_ratios, e2e_delays)

def run_experiment_2(env, agent):
    print("\n>>> RUNNING EXPERIMENT 2: Reconfigurability (Varying DCs: 2, 4, 6, 8)")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        current_config_completed = []
        current_config_dropped = []
        cpu_usages = []
        
        for ep in range(config.TEST_EPISODES):
            state, _ = env.reset(num_dcs=n_dc)
            action_mask = env._get_valid_actions_mask()
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon=config.TEST_EPSILON, valid_actions_mask=action_mask)
                state, _, done, _, info = env.step(action)
                action_mask = info.get('action_mask', None)
                
                total_cap = n_dc * config.DC_CPU_CYCLES
                curr_cap = sum(dc.cpu for dc in env.dcs)
                if total_cap > 0:
                    usage_pct = ((total_cap - curr_cap) / total_cap) * 100
                else:
                    usage_pct = 0
                cpu_usages.append(usage_pct)
            
            current_config_completed.extend(env.sfc_manager.completed_history)
            current_config_dropped.extend(env.sfc_manager.dropped_history)
                
            print(f"\r   > Config {n_dc} DCs: Ep {ep+1}/{config.TEST_EPISODES} done.", end="")
        
        avg_delay = np.mean([r.elapsed_time for r in current_config_completed]) if current_config_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        total_reqs = len(current_config_completed) + len(current_config_dropped)
        ar = (len(current_config_completed) / total_reqs * 100) if total_reqs > 0 else 0.0

        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        
        print(f"\n   -> Result: AR={ar:.2f}%, Delay={avg_delay:.2f}ms, CPU Usage={avg_cpu:.2f}%")

    plot_exp2(dc_counts, exp2_delays, exp2_resources)

def main():
    # 1. Init Env & Agent
    env = SFCNVEnv()
    agent = SFC_DQN()
    
    # 2. Load Weights - Try best model first, then fallback to regular
    weights_to_try = [f'models/best_{config.WEIGHTS_FILE}', f'models/{config.WEIGHTS_FILE}']
    
    loaded = False
    for weights_file in weights_to_try:
        if os.path.exists(weights_file):
            print(f"Loading weights from {weights_file}...")
            try:
                agent.model.load_weights(weights_file)
                print(f"✓ Weights loaded successfully from {weights_file}")
                loaded = True
                break
            except Exception as e:
                print(f"✗ Error loading {weights_file}: {e}")
    
    if not loaded:
        print(f"ERROR: No weights file found!")
        print(f"Please run 'python train.py' first to generate the model weights.")
        return

    # 3. Run Experiments
    run_experiment_1(env, agent)
    run_experiment_2(env, agent)
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    main()