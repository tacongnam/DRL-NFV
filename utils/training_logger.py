"""
Training Logger & Visualization Module
Tracks loss, rewards, and metrics across all training phases
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TrainingLogger:
    """Logs training metrics for VGAE, LL-Agent, and HL-Agent"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # VGAE Pre-training metrics
        self.vgae_losses: List[float] = []
        self.vgae_epochs: List[int] = []
        
        # LL-Agent Pre-training metrics
        self.ll_pretrain_losses: List[float] = []
        self.ll_pretrain_rewards: List[float] = []
        self.ll_pretrain_episodes: List[int] = []
        
        # HL-Agent Online Training metrics
        self.hl_losses_ar: List[float] = []  # Acceptance Ratio loss
        self.hl_losses_cost: List[float] = []  # Cost loss
        self.hl_steps: List[int] = []
        
        # LL-Agent Online Training metrics
        self.ll_losses: List[float] = []
        self.ll_weight_losses: List[float] = []
        self.ll_steps: List[int] = []
        
        # VGAE Online Training metrics
        self.vgae_online_losses: List[float] = []
        self.vgae_online_steps: List[int] = []
        
        # Episode metrics
        self.episode_acceptance_rates: List[float] = []
        self.episode_rewards: List[List[float]] = []
        self.episodes: List[int] = []
        
        self.start_time = datetime.now()
    
    def log_vgae_pretrain(self, epoch: int, loss: float):
        """Log VGAE pre-training loss"""
        self.vgae_epochs.append(epoch)
        self.vgae_losses.append(loss)
    
    def log_ll_pretrain(self, episode: int, loss: float, reward: float):
        """Log LL-Agent pre-training metrics"""
        self.ll_pretrain_episodes.append(episode)
        self.ll_pretrain_losses.append(loss)
        self.ll_pretrain_rewards.append(reward)
    
    def log_hl_train_step(self, step: int, loss_ar: float, loss_cost: float):
        """Log HL-Agent training step"""
        self.hl_steps.append(step)
        self.hl_losses_ar.append(loss_ar)
        self.hl_losses_cost.append(loss_cost)
    
    def log_ll_train_step(self, step: int, loss: float, weight_loss: float):
        """Log LL-Agent training step"""
        self.ll_steps.append(step)
        self.ll_losses.append(loss)
        self.ll_weight_losses.append(weight_loss)
    
    def log_vgae_online_train(self, step: int, loss: float):
        """Log VGAE online training"""
        self.vgae_online_steps.append(step)
        self.vgae_online_losses.append(loss)
    
    def log_episode(self, episode: int, acceptance_rate: float, rewards: List[float]):
        """Log episode statistics"""
        self.episodes.append(episode)
        self.episode_acceptance_rates.append(acceptance_rate)
        self.episode_rewards.append(rewards)
    
    def save_json(self, filename: str = "training_metrics.json"):
        """Save all metrics to JSON file"""
        data = {
            "timestamp": self.start_time.isoformat(),
            "vgae_pretrain": {
                "epochs": self.vgae_epochs,
                "losses": self.vgae_losses,
            },
            "ll_pretrain": {
                "episodes": self.ll_pretrain_episodes,
                "losses": self.ll_pretrain_losses,
                "rewards": self.ll_pretrain_rewards,
            },
            "hl_online_train": {
                "steps": self.hl_steps,
                "losses_ar": self.hl_losses_ar,
                "losses_cost": self.hl_losses_cost,
            },
            "ll_online_train": {
                "steps": self.ll_steps,
                "losses": self.ll_losses,
                "weight_losses": self.ll_weight_losses,
            },
            "vgae_online_train": {
                "steps": self.vgae_online_steps,
                "losses": self.vgae_online_losses,
            },
            "episodes": {
                "episodes": self.episodes,
                "acceptance_rates": self.episode_acceptance_rates,
            }
        }
        
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[TrainingLogger] Saved metrics → {path}")
        return path


class TrainingVisualizer:
    """Visualizes training metrics across all phases"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def load_metrics(self, filename: str = "training_metrics.json") -> Dict:
        """Load metrics from JSON file"""
        path = self.log_dir / filename
        if not path.exists():
            print(f"[TrainingVisualizer] File not found: {path}")
            return {}
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def plot_pretrain_phase(self, metrics: Dict, output_file: Optional[str] = None):
        """Plot pre-training phase (VGAE + LL-Agent)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Pre-training Phase", fontsize=16, fontweight='bold')
        
        # VGAE Loss
        vgae_data = metrics.get("vgae_pretrain", {})
        if vgae_data.get("epochs"):
            axes[0].plot(vgae_data["epochs"], vgae_data["losses"], 'b-o', linewidth=2, markersize=4)
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("VGAE Loss (BCE + KL)", fontsize=12)
            axes[0].set_title("VGAE Pre-training Loss", fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        
        # LL-Agent Loss & Reward
        ll_data = metrics.get("ll_pretrain", {})
        if ll_data.get("episodes"):
            ax1 = axes[1]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(ll_data["episodes"], ll_data["losses"], 'g-s', linewidth=2, markersize=4, label='DQN Loss')
            line2 = ax2.plot(ll_data["episodes"], ll_data["rewards"], 'r--^', linewidth=2, markersize=4, label='Avg Reward')
            
            ax1.set_xlabel("Episode", fontsize=12)
            ax1.set_ylabel("DQN Loss", fontsize=12, color='g')
            ax2.set_ylabel("Average Reward", fontsize=12, color='r')
            ax1.set_title("LL-Agent Pre-training", fontsize=13, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='g')
            ax2.tick_params(axis='y', labelcolor='r')
            ax1.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        if output_file:
            plt.savefig(self.log_dir / output_file, dpi=300, bbox_inches='tight')
            print(f"[TrainingVisualizer] Saved → {output_file}")
        plt.show()
    
    def plot_online_training_phase(self, metrics: Dict, output_file: Optional[str] = None):
        """Plot online training phase (HL + LL + VGAE)"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        fig.suptitle("Online Training Phase (High-Level & Low-Level Agents)", fontsize=16, fontweight='bold')
        
        # HL-Agent Loss
        hl_data = metrics.get("hl_online_train", {})
        ax1 = fig.add_subplot(gs[0, 0])
        if hl_data.get("steps"):
            ax1.plot(hl_data["steps"], hl_data["losses_ar"], 'b-', linewidth=1.5, alpha=0.7, label='Acceptance Loss')
            ax1.plot(hl_data["steps"], hl_data["losses_cost"], 'r--', linewidth=1.5, alpha=0.7, label='Cost Loss')
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.set_title("HL-Agent Losses", fontsize=12, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        
        # LL-Agent Losses
        ll_data = metrics.get("ll_online_train", {})
        ax2 = fig.add_subplot(gs[0, 1])
        if ll_data.get("steps"):
            ax2.plot(ll_data["steps"], ll_data["losses"], 'g-', linewidth=1.5, alpha=0.7, label='Q-network Loss')
            ax2.plot(ll_data["steps"], ll_data["weight_losses"], 'orange', linewidth=1.5, alpha=0.7, label='Weight Loss')
            ax2.set_ylabel("Loss", fontsize=11)
            ax2.set_title("LL-Agent Losses", fontsize=12, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        # VGAE Online Loss
        vgae_online = metrics.get("vgae_online_train", {})
        ax3 = fig.add_subplot(gs[1, :])
        if vgae_online.get("steps"):
            ax3.plot(vgae_online["steps"], vgae_online["losses"], 'purple', linewidth=1, alpha=0.7)
            ax3.fill_between(vgae_online["steps"], vgae_online["losses"], alpha=0.2, color='purple')
            ax3.set_ylabel("Loss", fontsize=11)
            ax3.set_title("VGAE Online Adaptation Loss", fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Episode Acceptance Rate
        episode_data = metrics.get("episodes", {})
        ax4 = fig.add_subplot(gs[2, :])
        if episode_data.get("episodes"):
            ax4.plot(episode_data["episodes"], episode_data["acceptance_rates"], 'darkgreen', 
                    linewidth=2, marker='o', markersize=4)
            ax4.fill_between(episode_data["episodes"], episode_data["acceptance_rates"], 
                            alpha=0.2, color='green')
            ax4.set_xlabel("Episode", fontsize=11)
            ax4.set_ylabel("Acceptance Rate", fontsize=11)
            ax4.set_title("Episode Acceptance Rate (Training Performance)", fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1.05])
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_file:
            plt.savefig(self.log_dir / output_file, dpi=300, bbox_inches='tight')
            print(f"[TrainingVisualizer] Saved → {output_file}")
        plt.show()
    
    def plot_all_phases(self, metrics: Dict, output_file: Optional[str] = None):
        """Plot all training phases in a comprehensive view"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("HRL-VGAE Complete Training Pipeline", fontsize=18, fontweight='bold', y=0.995)
        
        # ===== PRETRAIN PHASE =====
        # VGAE Pretrain
        ax1 = fig.add_subplot(gs[0, 0])
        vgae_data = metrics.get("vgae_pretrain", {})
        if vgae_data.get("epochs"):
            ax1.semilogy(vgae_data["epochs"], vgae_data["losses"], 'b-o', linewidth=2, markersize=3)
            ax1.set_ylabel("Loss", fontsize=10)
            ax1.set_title("VGAE Pre-training", fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        ax1.text(0.5, -0.3, "Phase 1: Offline Pre-training", ha='center', 
                transform=ax1.transAxes, fontsize=9, style='italic', color='gray')
        
        # LL Pretrain Loss
        ax2 = fig.add_subplot(gs[0, 1])
        ll_data = metrics.get("ll_pretrain", {})
        if ll_data.get("episodes"):
            ax2.plot(ll_data["episodes"], ll_data["losses"], 'g-s', linewidth=2, markersize=3)
            ax2.set_ylabel("Loss", fontsize=10)
            ax2.set_title("LL-Agent Pre-training Loss", fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # LL Pretrain Reward
        ax3 = fig.add_subplot(gs[0, 2])
        if ll_data.get("episodes"):
            ax3.plot(ll_data["episodes"], ll_data["rewards"], 'r--^', linewidth=2, markersize=3)
            ax3.set_ylabel("Reward", fontsize=10)
            ax3.set_title("LL-Agent Pre-training Reward", fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # ===== ONLINE TRAINING PHASE =====
        # HL-Agent Loss AR
        ax4 = fig.add_subplot(gs[1, 0])
        hl_data = metrics.get("hl_online_train", {})
        if hl_data.get("steps"):
            # Smooth the loss for better visualization
            losses_ar = np.array(hl_data["losses_ar"])
            window = max(1, len(losses_ar) // 50)
            smoothed = np.convolve(losses_ar, np.ones(window)/window, mode='valid')
            steps = hl_data["steps"][window-1:len(hl_data["steps"])]
            ax4.plot(steps, smoothed, 'b-', linewidth=2, alpha=0.8)
            ax4.plot(hl_data["steps"], losses_ar, 'b.', markersize=1, alpha=0.2)
            ax4.set_ylabel("Loss", fontsize=10)
            ax4.set_title("HL-Agent Acceptance Loss", fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        ax4.text(0.5, -0.3, "Phase 2: Online Training", ha='center',
                transform=ax4.transAxes, fontsize=9, style='italic', color='gray')
        
        # HL-Agent Loss Cost
        ax5 = fig.add_subplot(gs[1, 1])
        if hl_data.get("steps"):
            losses_cost = np.array(hl_data["losses_cost"])
            window = max(1, len(losses_cost) // 50)
            smoothed = np.convolve(losses_cost, np.ones(window)/window, mode='valid')
            steps = hl_data["steps"][window-1:len(hl_data["steps"])]
            ax5.plot(steps, smoothed, 'r-', linewidth=2, alpha=0.8)
            ax5.plot(hl_data["steps"], losses_cost, 'r.', markersize=1, alpha=0.2)
            ax5.set_ylabel("Loss", fontsize=10)
            ax5.set_title("HL-Agent Cost Loss", fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # HL Agent + LL Agent combined
        ax6 = fig.add_subplot(gs[1, 2])
        if hl_data.get("steps") and ll_data.get("episodes"):
            ax6_twin = ax6.twinx()
            line1 = ax6.plot(hl_data["steps"][:len(hl_data["losses_ar"])], 
                            hl_data["losses_ar"], 'b-', linewidth=1.5, alpha=0.6, label='HL Loss')
            line2 = ax6_twin.plot(hl_data["steps"][:len(ll_data.get("losses", []))], 
                                 ll_data.get("losses", [])[:len(hl_data["steps"])], 
                                 'g-', linewidth=1.5, alpha=0.6, label='LL Loss')
            ax6.set_ylabel("HL Loss", fontsize=10, color='b')
            ax6_twin.set_ylabel("LL Loss", fontsize=10, color='g')
            ax6.set_title("HL vs LL Training", fontsize=11, fontweight='bold')
            ax6.tick_params(axis='y', labelcolor='b')
            ax6_twin.tick_params(axis='y', labelcolor='g')
            ax6.grid(True, alpha=0.3)
        
        # LL-Agent Q-Loss
        ax7 = fig.add_subplot(gs[2, 0])
        ll_data = metrics.get("ll_online_train", {})
        if ll_data.get("steps"):
            losses = np.array(ll_data["losses"])
            window = max(1, len(losses) // 50)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            steps = ll_data["steps"][window-1:len(ll_data["steps"])]
            ax7.plot(steps, smoothed, 'g-', linewidth=2, alpha=0.8)
            ax7.plot(ll_data["steps"], losses, 'g.', markersize=1, alpha=0.2)
            ax7.set_ylabel("Loss", fontsize=10)
            ax7.set_title("LL-Agent Q-network Loss", fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # LL-Agent Weight Loss
        ax8 = fig.add_subplot(gs[2, 1])
        if ll_data.get("steps"):
            weight_losses = np.array(ll_data["weight_losses"])
            window = max(1, len(weight_losses) // 50)
            smoothed = np.convolve(weight_losses, np.ones(window)/window, mode='valid')
            steps = ll_data["steps"][window-1:len(ll_data["steps"])]
            ax8.plot(steps, smoothed, 'orange', linewidth=2, alpha=0.8)
            ax8.plot(ll_data["steps"], weight_losses, '.', color='orange', markersize=1, alpha=0.2)
            ax8.set_ylabel("Loss", fontsize=10)
            ax8.set_title("LL-Agent Weight Loss", fontsize=11, fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # VGAE Online Loss
        ax9 = fig.add_subplot(gs[2, 2])
        vgae_online = metrics.get("vgae_online_train", {})
        if vgae_online.get("steps"):
            losses = np.array(vgae_online["losses"])
            window = max(1, len(losses) // 50)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            steps = vgae_online["steps"][window-1:len(vgae_online["steps"])]
            ax9.plot(steps, smoothed, 'purple', linewidth=2, alpha=0.8)
            ax9.plot(vgae_online["steps"], losses, '.', color='purple', markersize=1, alpha=0.2)
            ax9.set_ylabel("Loss", fontsize=10)
            ax9.set_title("VGAE Online Adaptation", fontsize=11, fontweight='bold')
            ax9.grid(True, alpha=0.3)
        
        # ===== EPISODE METRICS =====
        ax10 = fig.add_subplot(gs[3, :])
        episode_data = metrics.get("episodes", {})
        if episode_data.get("episodes"):
            episodes = episode_data["episodes"]
            acc_rates = episode_data["acceptance_rates"]
            
            ax10.plot(episodes, acc_rates, 'darkgreen', linewidth=2.5, marker='o', markersize=4, label='Acceptance Rate')
            ax10.fill_between(episodes, acc_rates, alpha=0.2, color='green')
            
            # Add trend line
            if len(episodes) > 1:
                z = np.polyfit(episodes, acc_rates, 2)
                p = np.poly1d(z)
                ax10.plot(episodes, p(episodes), 'r--', linewidth=2, alpha=0.7, label='Trend')
            
            ax10.set_xlabel("Episode", fontsize=11)
            ax10.set_ylabel("Acceptance Rate", fontsize=11)
            ax10.set_title("Episode Performance (Acceptance Rate Improvement)", fontsize=12, fontweight='bold')
            ax10.set_ylim([0, 1.05])
            ax10.legend(loc='best', fontsize=10)
            ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_file:
            plt.savefig(self.log_dir / output_file, dpi=300, bbox_inches='tight')
            print(f"[TrainingVisualizer] Saved comprehensive plot → {output_file}")
        plt.show()
    
    def plot_convergence_summary(self, metrics: Dict, output_file: Optional[str] = None):
        """Plot convergence summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Convergence Summary", fontsize=16, fontweight='bold')
        
        # VGAE convergence
        ax = axes[0, 0]
        vgae_data = metrics.get("vgae_pretrain", {})
        if vgae_data.get("losses"):
            losses = np.array(vgae_data["losses"])
            ax.semilogy(range(len(losses)), losses, 'b-o', linewidth=2, markersize=3)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Loss (log scale)", fontsize=11)
            ax.set_title("VGAE Convergence", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Show improvement
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            ax.text(0.95, 0.05, f"Improvement: {improvement:.1f}%", 
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # LL-Agent Reward convergence
        ax = axes[0, 1]
        ll_data = metrics.get("ll_pretrain", {})
        if ll_data.get("rewards"):
            rewards = np.array(ll_data["rewards"])
            ax.plot(range(len(rewards)), rewards, 'g-^', linewidth=2, markersize=3)
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Average Reward", fontsize=11)
            ax.set_title("LL-Agent Reward Convergence", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if len(rewards) > 1:
                improvement = (rewards[-1] - rewards[0]) / max(abs(rewards[0]), 1) * 100
                ax.text(0.95, 0.05, f"Improvement: {improvement:+.1f}%",
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Online training loss comparison
        ax = axes[1, 0]
        hl_data = metrics.get("hl_online_train", {})
        ll_online = metrics.get("ll_online_train", {})
        if hl_data.get("losses_ar") and ll_online.get("losses"):
            hl_losses = np.array(hl_data["losses_ar"])
            ll_losses = np.array(ll_online["losses"][:len(hl_losses)])
            
            ax.plot(range(len(hl_losses)), hl_losses, 'b-', linewidth=1.5, label='HL Loss', alpha=0.7)
            ax.plot(range(len(ll_losses)), ll_losses, 'g-', linewidth=1.5, label='LL Loss', alpha=0.7)
            ax.set_xlabel("Step", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title("Online Training Loss Comparison", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Episode acceptance rate improvement
        ax = axes[1, 1]
        episode_data = metrics.get("episodes", {})
        if episode_data.get("acceptance_rates"):
            rates = np.array(episode_data["acceptance_rates"])
            ax.plot(range(len(rates)), rates, 'darkgreen', linewidth=2, marker='o', markersize=5)
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Acceptance Rate", fontsize=11)
            ax.set_title("Final Performance: Acceptance Rate", fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            
            if len(rates) > 1:
                final_rate = rates[-1]
                initial_rate = rates[0]
                ax.text(0.95, 0.05, f"Final: {final_rate:.3f} (Initial: {initial_rate:.3f})",
                       transform=ax.transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        if output_file:
            plt.savefig(self.log_dir / output_file, dpi=300, bbox_inches='tight')
            print(f"[TrainingVisualizer] Saved convergence summary → {output_file}")
        plt.show()


if __name__ == "__main__":
    # Example usage
    logger = TrainingLogger("training_logs")
    
    # Simulate VGAE pre-training
    for epoch in range(1, 51):
        loss = 10.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.1)
        logger.log_vgae_pretrain(epoch, max(loss, 0.1))
    
    # Simulate LL pre-training
    for ep in range(1, 51):
        loss = 5.0 * np.exp(-ep / 15) + np.random.normal(0, 0.05)
        reward = -loss + np.random.normal(0, 0.1)
        logger.log_ll_pretrain(ep, max(loss, 0.01), reward)
    
    # Simulate online training
    for step in range(1, 1001):
        hl_loss_ar = np.exp(-step / 500) + np.random.normal(0, 0.01)
        hl_loss_cost = 0.5 * np.exp(-step / 300) + np.random.normal(0, 0.005)
        ll_loss = 0.3 * np.exp(-step / 400) + np.random.normal(0, 0.005)
        ll_weight_loss = 0.2 * np.exp(-step / 350) + np.random.normal(0, 0.003)
        vgae_loss = 2.0 * np.exp(-step / 600) + np.random.normal(0, 0.02)
        
        logger.log_hl_train_step(step, max(hl_loss_ar, 0), max(hl_loss_cost, 0))
        logger.log_ll_train_step(step, max(ll_loss, 0), max(ll_weight_loss, 0))
        logger.log_vgae_online_train(step, max(vgae_loss, 0))
        
        if step % 50 == 0:
            acc_rate = 0.3 + 0.6 * (1 - np.exp(-step / 500)) + np.random.normal(0, 0.05)
            logger.log_episode(step // 50, np.clip(acc_rate, 0, 1), [acc_rate])
    
    # Save metrics
    logger.save_json()
    
    # Visualize
    visualizer = TrainingVisualizer("training_logs")
    metrics = visualizer.load_metrics()
    
    visualizer.plot_pretrain_phase(metrics, "01_pretrain_phase.png")
    visualizer.plot_online_training_phase(metrics, "02_online_training.png")
    visualizer.plot_all_phases(metrics, "03_complete_pipeline.png")
    visualizer.plot_convergence_summary(metrics, "04_convergence_summary.png")
