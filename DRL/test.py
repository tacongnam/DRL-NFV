import numpy as np
import matplotlib.pyplot as plt
import os
import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN

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
    
    # Set y limit for delay comfortably above max delay
    max_delay = max(delays) if delays else 100
    ax2.set_ylim(0, max_delay * 1.2)
    
    plt.title('Experiment 1: Performance per SFC Type (Fixed 4 DCs)')
    fig.tight_layout()
    plt.savefig('result_exp1_fig2.png')
    print("Saved result_exp1_fig2.png")
    # plt.show() # Uncomment nếu chạy trên môi trường có GUI

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
    plt.savefig('result_exp2_fig3.png')
    print("Saved result_exp2_fig3.png")
    # plt.show()

def run_experiment_1(env, agent):
    print("\n>>> RUNNING EXPERIMENT 1: Performance Analysis per SFC Type (4 DCs)")
    
    # Các biến tích lũy toàn bộ kết quả của Experiment 1
    total_completed_history = []
    total_dropped_history = []
    
    for ep in range(config.TEST_EPISODES):
        # Reset môi trường
        state, _ = env.reset(num_dcs=4)
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            action = agent.get_action(state, epsilon=config.TEST_EPSILON)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
        # --- QUAN TRỌNG: Lưu lại history trước khi env.reset() xóa nó ở vòng lặp sau ---
        total_completed_history.extend(env.sfc_manager.completed_history)
        total_dropped_history.extend(env.sfc_manager.dropped_history)
        
        print(f"\r   > Ep {ep+1}/{config.TEST_EPISODES} completed. Reward: {total_reward:.1f}. "
              f"Requests (Done/Drop): {len(env.sfc_manager.completed_history)}/{len(env.sfc_manager.dropped_history)}", end="")
    print("")

    # Tổng hợp số liệu từ danh sách tích lũy
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    print("-" * 75)
    print(f"{'SFC Type':<15} | {'Acc Ratio (%)':<15} | {'Avg Delay (ms)':<15} | {'Count (Done/All)'}")
    print("-" * 75)
    
    for s_type in sfc_types:
        # Lọc từ danh sách tổng
        completed = [r for r in total_completed_history if r.type == s_type]
        dropped = [r for r in total_dropped_history if r.type == s_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.elapsed_time for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        print(f"{s_type:<15} | {ar:<15.2f} | {avg_delay:<15.2f} | {len(completed)}/{total}")
    
    plot_exp1(sfc_types, acc_ratios, e2e_delays)

def run_experiment_2(env, agent):
    print("\n>>> RUNNING EXPERIMENT 2: Reconfigurability (Varying DCs: 2, 4, 6, 8)")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        # Biến tích lũy cho cấu hình DC hiện tại
        current_config_completed = []
        current_config_dropped = []
        cpu_usages = []
        
        for ep in range(config.TEST_EPISODES):
            state, _ = env.reset(num_dcs=n_dc)
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon=config.TEST_EPSILON)
                state, _, done, _, _ = env.step(action)
                
                # Thu thập thông tin tài nguyên tại mỗi bước
                total_cap = n_dc * config.DC_CPU_CYCLES
                curr_cap = sum(dc.cpu for dc in env.dcs)
                # Tránh chia cho 0
                if total_cap > 0:
                    usage_pct = ((total_cap - curr_cap) / total_cap) * 100
                else:
                    usage_pct = 0
                cpu_usages.append(usage_pct)
            
            # Tích lũy history
            current_config_completed.extend(env.sfc_manager.completed_history)
            current_config_dropped.extend(env.sfc_manager.dropped_history)
                
            print(f"\r   > Config {n_dc} DCs: Ep {ep+1}/{config.TEST_EPISODES} done.", end="")
        
        # Tính toán trung bình cho cấu hình này
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
    
    # 2. Load Weights
    if os.path.exists(config.WEIGHTS_FILE):
        print(f"Loading weights from {config.WEIGHTS_FILE}...")
        try:
            agent.model.load_weights(config.WEIGHTS_FILE)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Make sure you are running in the same directory as the weights file.")
            return
    else:
        print(f"ERROR: Weights file '{config.WEIGHTS_FILE}' not found!")
        print("Please run 'python train.py' first to generate the model weights.")
        return

    # 3. Run Experiments
    run_experiment_1(env, agent)
    run_experiment_2(env, agent)
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    main()