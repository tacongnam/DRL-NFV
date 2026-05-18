import os
import json
from collections import defaultdict


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.data = defaultdict(list)

    def log_vgae_pretrain(self, epoch: int, loss: float):
        self.data["vgae_pretrain_epoch"].append(epoch)
        self.data["vgae_pretrain_loss"].append(float(loss))

    def log_ll_pretrain(self, episode: int, loss: float, avg_reward: float):
        self.data["ll_pretrain_episode"].append(episode)
        self.data["ll_pretrain_loss"].append(float(loss))
        self.data["ll_pretrain_avg_reward"].append(float(avg_reward))

    def log_episode(self, episode: int, acceptance_ratio: float, counts: list):
        self.data["train_episode"].append(episode)
        self.data["train_acceptance_ratio"].append(float(acceptance_ratio))
        self.data["train_accepted"].append(int(counts[0]))
        self.data["train_rejected"].append(int(counts[1]))

    def log_hl_train_step(self, step: int, loss_ar: float, loss_cost: float):
        self.data["hl_step"].append(step)
        self.data["hl_loss_ar"].append(float(loss_ar))
        self.data["hl_loss_cost"].append(float(loss_cost))

    def log_ll_train_step(self, step: int, loss: float, weight_loss: float):
        self.data["ll_step"].append(step)
        self.data["ll_loss"].append(float(loss))
        self.data["ll_weight_loss"].append(float(weight_loss))

    def log_vgae_online_train(self, step: int, loss: float):
        self.data["vgae_online_step"].append(step)
        self.data["vgae_online_loss"].append(float(loss))

    def save(self):
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(dict(self.data), f)
        return path

    def plot_learning_curves(self, out_dir: str = None):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[Logger] matplotlib not available, skipping plots.")
            return

        save_dir = out_dir or self.log_dir
        os.makedirs(save_dir, exist_ok=True)

        def smooth(vals, w=5):
            if len(vals) < w:
                return vals
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        def save_fig(fig, name):
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Logger] Saved → {path}")

        if self.data.get("vgae_pretrain_epoch"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.data["vgae_pretrain_epoch"],
                    self.data["vgae_pretrain_loss"], color="#4c7be0", linewidth=1.5)
            ax.set_title("VGAE Pre-training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            save_fig(fig, "pretrain_vgae_loss.png")

        if self.data.get("ll_pretrain_episode"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].plot(self.data["ll_pretrain_episode"],
                         self.data["ll_pretrain_avg_reward"],
                         color="#e07b4c", linewidth=1.5)
            axes[0].set_title("LL Pre-training Avg Reward")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Avg Reward")
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].plot(self.data["ll_pretrain_episode"],
                         self.data["ll_pretrain_loss"],
                         color="#c94040", linewidth=1.5)
            axes[1].set_title("LL Pre-training Loss")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            save_fig(fig, "pretrain_ll_curves.png")

        if self.data.get("train_episode"):
            fig, ax = plt.subplots(figsize=(9, 4))
            ar = self.data["train_acceptance_ratio"]
            eps = self.data["train_episode"]
            ax.plot(eps, ar, color="#aaaaaa", linewidth=0.8, alpha=0.5, label="Raw")
            s = smooth(ar, w=max(3, len(ar) // 10))
            ax.plot(eps[:len(s)], s, color="#4c7be0", linewidth=2.0, label="Smoothed")
            ax.set_title("HRL Training — Acceptance Ratio per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Acceptance Ratio")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            save_fig(fig, "train_acceptance_ratio.png")

        if self.data.get("hl_step"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for ax, key, label, color in [
                (axes[0], "hl_loss_ar",   "HL Loss (AR)",   "#5b8dd9"),
                (axes[1], "hl_loss_cost", "HL Loss (Cost)", "#e05c5c"),
            ]:
                vals = self.data[key]
                steps = self.data["hl_step"]
                ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.5)
                s = smooth(vals)
                ax.plot(steps[:len(s)], s, color=color, linewidth=2.0)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle="--", alpha=0.5)
            fig.suptitle("High-Level Agent Training Loss", fontweight="bold")
            fig.tight_layout()
            save_fig(fig, "train_hl_loss.png")

        if self.data.get("ll_step"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for ax, key, label, color in [
                (axes[0], "ll_loss",        "LL Policy Loss",  "#4ca87e"),
                (axes[1], "ll_weight_loss", "LL Weight Loss",  "#9b59b6"),
            ]:
                vals = self.data[key]
                steps = self.data["ll_step"]
                ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.5)
                s = smooth(vals)
                ax.plot(steps[:len(s)], s, color=color, linewidth=2.0)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle="--", alpha=0.5)
            fig.suptitle("Low-Level Agent Training Loss", fontweight="bold")
            fig.tight_layout()
            save_fig(fig, "train_ll_loss.png")

        if self.data.get("vgae_online_step"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.data["vgae_online_step"],
                    self.data["vgae_online_loss"], color="#27ae60", linewidth=1.2)
            ax.set_title("VGAE Online Training Loss (during HRL)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            save_fig(fig, "train_vgae_online_loss.png")