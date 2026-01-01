import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png'):
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

def plot_overall_results(acceptance_ratios, avg_delays, throughputs, save_path='fig/result_overall.png'):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Acceptance Ratio Distribution
        axes[0].hist(acceptance_ratios, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(acceptance_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(acceptance_ratios):.2f}%')
        axes[0].set_xlabel('Acceptance Ratio (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Acceptance Ratio Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: E2E Delay Distribution
        axes[1].hist(avg_delays, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(avg_delays), color='red', linestyle='--',
                       label=f'Mean: {np.mean(avg_delays):.2f} ms')
        axes[1].set_xlabel('Avg E2E Delay (ms)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('E2E Delay Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Throughput Distribution
        axes[2].hist(throughputs, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(throughputs), color='red', linestyle='--',
                       label=f'Mean: {np.mean(throughputs):.2f} Mbps')
        axes[2].set_xlabel('Throughput (Mbps)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Throughput Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"\n[Error] Could not plot overall results: {e}")

def plot_scalability_results(results, save_path='fig/result_scalability.png'):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        dc_counts = results['dc_counts']
        
        # Plot 1: Acceptance Ratio vs DC Count
        axes[0].plot(dc_counts, results['acceptance_ratios'], 'b-o', linewidth=2)
        axes[0].set_xlabel('Number of DCs')
        axes[0].set_ylabel('Acceptance Ratio (%)')
        axes[0].set_title('Acceptance Ratio vs DC Count')
        axes[0].set_xticks(dc_counts)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: E2E Delay vs DC Count
        axes[1].plot(dc_counts, results['avg_delays'], 'g-o', linewidth=2)
        axes[1].set_xlabel('Number of DCs')
        axes[1].set_ylabel('Avg E2E Delay (ms)')
        axes[1].set_title('E2E Delay vs DC Count')
        axes[1].set_xticks(dc_counts)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: CPU Usage vs DC Count
        axes[2].bar(dc_counts, results['cpu_usages'], color='orange', alpha=0.7, width=0.6)
        axes[2].set_xlabel('Number of DCs')
        axes[2].set_ylabel('Avg CPU Usage (%)')
        axes[2].set_title('Resource Utilization vs DC Count')
        axes[2].set_xticks(dc_counts)
        axes[2].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"\n[Error] Could not plot scalability: {e}")