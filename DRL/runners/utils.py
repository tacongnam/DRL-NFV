import os
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png'):
    """Vẽ biểu đồ quá trình huấn luyện"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Acceptance Ratio
        episodes = list(range(1, len(all_ars) + 1))
        ax1.plot(episodes, all_ars, alpha=0.3, label='Per Episode')
        
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
        ax2.plot(episodes, all_rewards, alpha=0.3, label='Per Episode')
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
        print(f"Training progress plot saved to: {save_path}")
        plt.close(fig) # Close to free memory
    except Exception as e:
        print(f"Could not create training plot: {e}")

def plot_exp1_results(types, ars, delays, save_path='fig/result_exp1_fig2.png'):
    """Vẽ biểu đồ Experiment 1"""
    if not types:
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close(fig)

def plot_exp2_results(dc_counts, delays, resources, save_path='fig/result_exp2_fig3.png'):
    """Vẽ biểu đồ Experiment 2"""
    if not dc_counts:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close(fig)


# --- HELPER FUNCTIONS FOR LOGIC ---

def run_single_episode(env, agent, epsilon, training_mode=True):
    """
    Chạy 1 episode hoàn chỉnh.
    Args:
        training_mode: Nếu True, trả về memory trace. Nếu False, chỉ chạy để test.
    Returns:
        total_reward, acc_ratio, memory_trace (optional)
    """
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0
    done = False
    episode_memory = []
    
    while not done:
        action = agent.get_action(state, epsilon, valid_actions_mask=action_mask)
        next_state, reward, done, _, info = env.step(action)
        
        # Lấy mask mới cho step tiếp theo
        next_action_mask = info.get('action_mask', None)
        
        if training_mode:
            # Lưu transition: (state, action, reward, next_state, done)
            episode_memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        action_mask = next_action_mask

    # Lấy thông tin thống kê cuối episode
    acc_ratio = info.get('acc_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio

def run_experiment_performance(env, agent, episodes=10):
    """Logic chạy Experiment 1"""
    print(f"\n>>> RUNNING EXPERIMENT 1: Performance Analysis per SFC Type (4 DCs)")
    
    total_completed = []
    total_dropped = []
    
    for ep in range(episodes):
        env.reset(num_dcs=4)
        run_single_episode(env, agent, epsilon=config.TEST_EPSILON, training_mode=False)
        
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
        print(f"\r   > Ep {ep+1}/{episodes} done.", end="")
    
    print("\n   Processing results...")
    
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    for s_type in sfc_types:
        completed = [r for r in total_completed if r.type == s_type]
        dropped = [r for r in total_dropped if r.type == s_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.elapsed_time for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        print(f"    Type {s_type}: AR={ar:.1f}%, Delay={avg_delay:.1f}ms")
        
    plot_exp1_results(sfc_types, acc_ratios, e2e_delays)

def run_experiment_scalability(env, agent, episodes=10):
    """Logic chạy Experiment 2"""
    print(f"\n>>> RUNNING EXPERIMENT 2: Reconfigurability")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        current_completed = []
        cpu_usages = []
        
        for ep in range(episodes):
            env.reset(num_dcs=n_dc)
            state, _ = env._get_state()
            action_mask = env._get_valid_actions_mask()
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon=config.TEST_EPSILON, valid_actions_mask=action_mask)
                state, _, done, _, info = env.step(action)
                action_mask = info.get('action_mask', None)
                
                # Tính CPU usage tức thời
                total_cap = n_dc * config.DC_CPU_CYCLES
                curr_cap = sum(dc.cpu for dc in env.dcs)
                usage_pct = ((total_cap - curr_cap) / total_cap) * 100 if total_cap > 0 else 0
                cpu_usages.append(usage_pct)
            
            current_completed.extend(env.sfc_manager.completed_history)
            print(f"\r   > Config {n_dc} DCs: Ep {ep+1}/{episodes}", end="")
            
        avg_delay = np.mean([r.elapsed_time for r in current_completed]) if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        print(f" -> Delay={avg_delay:.1f}ms, CPU={avg_cpu:.1f}%")

    plot_exp2_results(dc_counts, exp2_delays, exp2_resources)