import os
import json
from collections import defaultdict


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir   = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.data      = defaultdict(list)
        self._global_step = 0
        self._file_boundaries: list = []

    def info(self, msg, *args):
        if args:
            msg = msg % args
        print(f"[INFO] {msg}")

    def warning(self, msg, *args):
        if args:
            msg = msg % args
        print(f"[WARN] {msg}")

    def error(self, msg, *args):
        if args:
            msg = msg % args
        print(f"[ERROR] {msg}")

    def mark_file_boundary(self, file_name: str):
        self._file_boundaries.append({
            "step": self._global_step,
            "file": file_name,
        })
        self.data["file_boundaries"] = self._file_boundaries

    def log_vgae_pretrain(self, epoch: int, loss: float):
        self.data["vgae_pretrain_epoch"].append(epoch)
        self.data["vgae_pretrain_loss"].append(float(loss))

    def log_ll_pretrain(self, episode: int, loss: float, avg_reward: float):
        self.data["ll_pretrain_episode"].append(episode)
        self.data["ll_pretrain_loss"].append(float(loss))
        self.data["ll_pretrain_avg_reward"].append(float(avg_reward))

    def log_episode(self, episode: int, acceptance_ratio: float, counts: list):
        self._global_step += 1
        self.data["train_episode"].append(episode)
        self.data["train_global_step"].append(self._global_step)
        self.data["train_acceptance_ratio"].append(float(acceptance_ratio))
        self.data["train_accepted"].append(int(counts[0]))
        self.data["train_rejected"].append(int(counts[1]))

    def log_vgae_finetune(self, step: int, loss: float):
        self.data["vgae_finetune_step"].append(step)
        self.data["vgae_finetune_loss"].append(float(loss))

    def log_vgae_online_train(self, step: int, loss: float):
        self.data["vgae_online_step"].append(step)
        self.data["vgae_online_loss"].append(float(loss))

    def log_hl_train_step(self, step: int, loss_ar: float, loss_cost: float):
        self.data["hl_step"].append(step)
        self.data["hl_loss_ar"].append(float(loss_ar))
        self.data["hl_loss_cost"].append(float(loss_cost))

    def log_ll_train_step(self, step: int, loss: float, weight_loss: float):
        self.data["ll_step"].append(step)
        self.data["ll_loss"].append(float(loss))
        self.data["ll_weight_loss"].append(float(weight_loss))

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

        boundaries = self.data.get("file_boundaries", [])

        def smooth(vals, w=5):
            if len(vals) < w:
                return np.array(vals, dtype=float)
            return np.convolve(vals, np.ones(w) / w, mode="valid")

        def draw_boundaries(ax, x_vals, boundaries, color="#aaaaaa"):
            for b in boundaries:
                bx = b["step"]
                if len(x_vals) > 0 and x_vals[0] <= bx <= x_vals[-1]:
                    ax.axvline(x=bx, color=color, linewidth=0.8,
                               linestyle="--", alpha=0.7)
                    ax.text(bx, ax.get_ylim()[1] * 0.97,
                            os.path.basename(b["file"]),
                            fontsize=5, rotation=90,
                            va="top", ha="right", color=color)

        def save_fig(fig, name):
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Logger] Saved → {path}")

        # VGAE pretrain loss
        if self.data.get("vgae_pretrain_epoch"):
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs = self.data["vgae_pretrain_epoch"]
            losses = self.data["vgae_pretrain_loss"]
            ax.plot(epochs, losses, color="#bbbbbb", linewidth=0.8, alpha=0.5, label="Raw")
            s = smooth(losses, w=max(3, len(losses) // 10))
            ax.plot(epochs[:len(s)], s, color="#4c7be0", linewidth=1.8, label="Smoothed")
            ax.set_title("VGAE Pre-training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, "pretrain_vgae_loss.png")

        # LL pretrain
        if self.data.get("ll_pretrain_episode"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            eps    = self.data["ll_pretrain_episode"]
            reward = self.data["ll_pretrain_avg_reward"]
            loss   = self.data["ll_pretrain_loss"]

            for ax, vals, title, color in [
                (axes[0], reward, "LL Pre-training Avg Reward", "#e07b4c"),
                (axes[1], loss,   "LL Pre-training Loss",       "#c94040"),
            ]:
                ax.plot(eps, vals, color="#dddddd", linewidth=0.8, alpha=0.5)
                s = smooth(vals, w=max(3, len(vals) // 10))
                ax.plot(eps[:len(s)], s, color=color, linewidth=1.8)
                ax.set_title(title)
                ax.set_xlabel("Episode")
                ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, "pretrain_ll_curves.png")

        # Training acceptance ratio — dùng global_step, vẽ boundary
        if self.data.get("train_episode"):
            fig, ax = plt.subplots(figsize=(10, 4))
            x_key = "train_global_step" if self.data.get("train_global_step") else "train_episode"
            xs = self.data[x_key]
            ar = self.data["train_acceptance_ratio"]
            ax.plot(xs, ar, color="#cccccc", linewidth=0.6, alpha=0.4, label="Raw")
            w  = max(3, len(ar) // 15)
            s  = smooth(ar, w=w)
            ax.plot(xs[:len(s)], s, color="#4c7be0", linewidth=2.0, label=f"Smoothed (w={w})")
            draw_boundaries(ax, xs, boundaries)
            ax.set_title("HRL Training — Acceptance Ratio")
            ax.set_xlabel("Global Episode" if x_key == "train_global_step" else "Episode")
            ax.set_ylabel("Acceptance Ratio")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, "train_acceptance_ratio.png")

        # VGAE fine-tune loss
        if self.data.get("vgae_finetune_step"):
            fig, ax = plt.subplots(figsize=(8, 4))
            steps  = self.data["vgae_finetune_step"]
            losses = self.data["vgae_finetune_loss"]
            ax.plot(steps, losses, color="#bbbbbb", linewidth=0.6, alpha=0.5)
            s = smooth(losses, w=max(3, len(losses) // 10))
            ax.plot(steps[:len(s)], s, color="#27ae60", linewidth=1.8)
            draw_boundaries(ax, steps, boundaries)
            ax.set_title("VGAE Fine-tune Loss (aux head only)")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Aux Loss")
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, "train_vgae_finetune_loss.png")

        # HL loss
        if self.data.get("hl_step"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for ax, key, label, color in [
                (axes[0], "hl_loss_ar",   "HL Loss (AR)",   "#5b8dd9"),
                (axes[1], "hl_loss_cost", "HL Loss (Cost)", "#e05c5c"),
            ]:
                vals  = self.data[key]
                steps = self.data["hl_step"]
                ax.plot(steps, vals, color="#dddddd", linewidth=0.6, alpha=0.5)
                s = smooth(vals)
                ax.plot(steps[:len(s)], s, color=color, linewidth=2.0)
                draw_boundaries(ax, steps, boundaries)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle="--", alpha=0.4)
            fig.suptitle("High-Level Agent Training Loss", fontweight="bold")
            fig.tight_layout()
            save_fig(fig, "train_hl_loss.png")

        # LL loss
        if self.data.get("ll_step"):
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for ax, key, label, color in [
                (axes[0], "ll_loss",        "LL Policy Loss", "#4ca87e"),
                (axes[1], "ll_weight_loss", "LL Weight Loss", "#9b59b6"),
            ]:
                vals  = self.data[key]
                steps = self.data["ll_step"]
                ax.plot(steps, vals, color="#dddddd", linewidth=0.6, alpha=0.5)
                s = smooth(vals)
                ax.plot(steps[:len(s)], s, color=color, linewidth=2.0)
                draw_boundaries(ax, steps, boundaries)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.grid(True, linestyle="--", alpha=0.4)
            fig.suptitle("Low-Level Agent Training Loss", fontweight="bold")
            fig.tight_layout()
            save_fig(fig, "train_ll_loss.png")

        # VGAE online loss (legacy)
        if self.data.get("vgae_online_step"):
            fig, ax = plt.subplots(figsize=(8, 4))
            steps  = self.data["vgae_online_step"]
            losses = self.data["vgae_online_loss"]
            ax.plot(steps, losses, color="#bbbbbb", linewidth=0.6, alpha=0.5)
            s = smooth(losses)
            ax.plot(steps[:len(s)], s, color="#27ae60", linewidth=1.8)
            draw_boundaries(ax, steps, boundaries)
            ax.set_title("VGAE Online Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, "train_vgae_online_loss.png")