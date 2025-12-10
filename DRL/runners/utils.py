# runners/utils.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

def run_single_episode(env, agent, epsilon, training_mode=True):
    """
    Chạy một episode
    
    Args:
        env: Environment
        agent: DQN Agent
        epsilon: Exploration rate
        training_mode: Nếu True, lưu transitions vào memory
        
    Returns:
        Nếu training_mode: (total_reward, acceptance_ratio, episode_memory)
        Nếu không: (total_reward, acceptance_ratio)
    """
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0.0
    done = False
    episode_memory = []
    step_count = 0
    
    while not done:
        # Select action
        action = agent.get_action(state, epsilon, valid_actions_mask=action_mask)
        
        # Take step
        next_state, reward, done, _, info = env.step(action)
        next_action_mask = info.get('action_mask', None)
        
        # Store transition
        if training_mode:
            episode_memory.append((state, action, reward, next_state, done))
        
        # Update
        state = next_state
        total_reward += reward
        action_mask = next_action_mask
        
        # Progress indicator for testing
        step_count += 1
        if not training_mode and step_count % 500 == 0:
            print(".", end="", flush=True)
    
    acc_ratio = info.get('acceptance_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio

def plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png'):
    """Vẽ biểu đồ quá trình training"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        episodes = list(range(1, len(all_ars) + 1))
        
        # Plot 1: Acceptance Ratio
        ax1.plot(episodes, all_ars, alpha=0.3, label='Per Episode', color='blue')
        
        window = 20
        if len(all_ars) >= window:
            moving_avg = np.convolve(all_ars, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(all_ars) + 1), moving_avg,
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Acceptance Ratio (%)')
        ax1.set_title('Training Progress: Acceptance Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total Reward
        ax2.plot(episodes, all_rewards, alpha=0.3, label='Per Episode', color='green')
        
        if len(all_rewards) >= window:
            moving_avg_rew = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(all_rewards) + 1), moving_avg_rew,
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Training Progress: Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Training progress saved to: {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not create training plot: {e}")

def plot_exp1_results(sfc_types, acc_ratios, e2e_delays, save_path='fig/result_exp1_fig2.png'):
    """Vẽ biểu đồ Experiment 1"""
    if not sfc_types:
        return
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sfc_types))
        width = 0.35
        
        # Bar: Acceptance Ratio
        ax1.bar(x, acc_ratios, width, label='Acceptance Ratio (%)', 
               color='b', alpha=0.6)
        ax1.set_xlabel('SFC Types')
        ax1.set_ylabel('Acceptance Ratio (%)', color='b')
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sfc_types, rotation=15)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Line: E2E Delay
        ax2 = ax1.twinx()
        ax2.plot(x, e2e_delays, color='r', marker='o', 
                linewidth=2, label='E2E Delay (ms)')
        ax2.set_ylabel('Avg E2E Delay (ms)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        max_delay = max(e2e_delays) if e2e_delays else 100
        ax2.set_ylim(0, max_delay * 1.2)
        
        plt.title('Experiment 1: Performance per SFC Type (Fixed 4 DCs)')
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not plot Exp1: {e}")

def plot_exp2_results(dc_counts, delays, resources, save_path='fig/result_exp2_fig3.png'):
    """Vẽ biểu đồ Experiment 2"""
    if not dc_counts:
        return
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graph 1: E2E Delay
        ax1.plot(dc_counts, delays, 'g-o', linewidth=2)
        ax1.set_title('E2E Delay vs Number of DCs')
        ax1.set_xlabel('Number of DCs')
        ax1.set_ylabel('Avg E2E Delay (ms)')
        ax1.set_xticks(dc_counts)
        ax1.grid(True)
        
        # Graph 2: Resource Consumption
        ax2.bar(dc_counts, resources, color='orange', alpha=0.7, width=0.8)
        ax2.set_title('Avg Resource Consumption vs Number of DCs')
        ax2.set_xlabel('Number of DCs')
        ax2.set_ylabel('Avg CPU Usage (%)')
        ax2.set_xticks(dc_counts)
        ax2.grid(True, axis='y')
        
        plt.suptitle('Experiment 2: Reconfigurability & Robustness')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not plot Exp2: {e}")

def run_experiment_performance(env, agent, episodes=10):
    """Chạy Experiment 1: Performance Analysis"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Performance Analysis per SFC Type (4 DCs)")
    print(f"{'='*80}")
    
    total_completed = []
    total_dropped = []
    
    for ep in range(episodes):
        print(f"\n[Episode {ep+1}/{episodes}] Running", end=" ", flush=True)
        env.reset(num_dcs=4)
        run_single_episode(env, agent, epsilon=config.TEST_EPSILON, training_mode=False)
        
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
        print(" ✓")
    
    print("\nProcessing results...")
    
    # Analyze per SFC type
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
    """Chạy Experiment 2: Reconfigurability"""
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
            print(f"  Episode {ep+1}/{episodes} running", end=" ", flush=True)
            env.reset(num_dcs=n_dc)
            
            state, _ = env._get_obs(), {}
            done = False
            step_count = 0
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=config.TEST_EPSILON, 
                                         valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                # Track CPU usage
                total_cap = n_dc * config.DC_CPU_CYCLES
                used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in env.dcs)
                usage_pct = (used_cap / total_cap * 100) if total_cap > 0 else 0
                cpu_usages.append(usage_pct)
                
                step_count += 1
                if step_count % 500 == 0:
                    print(".", end="", flush=True)
            
            current_completed.extend(env.sfc_manager.completed_history)
            print(" ✓")
        
        avg_delay = np.mean([r.get_total_e2e_delay() for r in current_completed]) if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        
        print(f"  → Avg E2E Delay: {avg_delay:.2f} ms  |  Avg CPU Usage: {avg_cpu:.2f}%")
    
    plot_exp2_results(dc_counts, exp2_delays, exp2_resources)