import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    if not sfc_types: return
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sfc_types))
        width = 0.35
        
        ax1.bar(x, acc_ratios, width, label='Acceptance Ratio (%)', color='b', alpha=0.6)
        ax1.set_xlabel('SFC Types')
        ax1.set_ylabel('Acceptance Ratio (%)', color='b')
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sfc_types, rotation=15)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.plot(x, e2e_delays, color='r', marker='o', linewidth=2, label='E2E Delay (ms)')
        ax2.set_ylabel('Avg E2E Delay (ms)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        max_delay = max(e2e_delays) if e2e_delays else 100
        ax2.set_ylim(0, max_delay * 1.2)
        
        plt.title('Experiment 1: Performance per SFC Type')
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"\n[Error] Could not plot Exp1: {e}")

def plot_exp2_results(dc_counts, delays, resources, save_path='fig/result_exp2_fig3.png'):
    """Vẽ biểu đồ Experiment 2"""
    if not dc_counts: return
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(dc_counts, delays, 'g-o', linewidth=2)
        ax1.set_title('E2E Delay vs Number of DCs')
        ax1.set_xlabel('Number of DCs')
        ax1.set_ylabel('Avg E2E Delay (ms)')
        ax1.set_xticks(dc_counts)
        ax1.grid(True)
        
        ax2.bar(dc_counts, resources, color='orange', alpha=0.7, width=0.8)
        ax2.set_title('Avg Resource Consumption')
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