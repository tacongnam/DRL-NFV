from __future__ import annotations
import collections
import os
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from keras import layers
import config

# ── Kích thước state ──────────────────────────────────────────
# Gp : mean available ratio (3) + mean load (3) + mean MM1 (1) + mean link load (1) = 8
# Gq : mean VNF resource (3) + bw (1) + num_vnfs normalized (1)                     = 5
# Oq : revenue normalized (1) + cost normalized (1) + R2C normalized (1)             = 3
GP_DIM      = 8
GQ_DIM      = 5
OQ_DIM      = 3
TRIPLET_DIM = GP_DIM + GQ_DIM + OQ_DIM     # 16
WINDOW_SIZE = 10                             # N triplets lịch sử


class AdmissionAgent:
    """
    Refine-level agent: binary accept/reject dựa trên lịch sử N state triplets.
    Architecture: GRU(window) → actor (binary logits) + critic (scalar value)
    Training: PPO với GAE, update cuối mỗi episode.
    """

    def __init__(self, triplet_dim: int = TRIPLET_DIM,
                 window_size: int = WINDOW_SIZE,
                 hidden_dim: int = 64, lr: float = 1e-3):
        self.triplet_dim = triplet_dim
        self.window_size = window_size
        self.hidden_dim  = hidden_dim

        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._traj:    list              = []   # (window, action, log_prob, value, reward)

        self.actor  = self._build_rnn_model("admission_actor",  out_dim=2)
        self.critic = self._build_rnn_model("admission_critic", out_dim=1)

        self.opt_actor  = keras.optimizers.Adam(lr)
        self.opt_critic = keras.optimizers.Adam(lr)

        # Warm-up call để build weights
        _dummy = np.zeros((1, window_size, triplet_dim), np.float32)
        self.actor(_dummy)
        self.critic(_dummy)

    def _build_rnn_model(self, name: str, out_dim: int) -> keras.Model:
        inp = keras.Input(shape=(self.window_size, self.triplet_dim), name=name + "_in")
        x   = layers.GRU(self.hidden_dim, return_sequences=False)(inp)
        x   = layers.Dense(self.hidden_dim // 2, activation="relu")(x)
        out = layers.Dense(out_dim)(x)
        return keras.Model(inp, out, name=name)

    # ── Feature extraction ────────────────────────────────────

    @staticmethod
    def extract_gp_features(env, t_start: int, t_end: int) -> np.ndarray:
        """8-dim snapshot của network state tại [t_start, t_end]"""
        dc_nodes = [n for n in env.network.nodes.values()
                    if n.type == config.NODE_DC and n.cap]
        if not dc_nodes:
            return np.zeros(GP_DIM, np.float32)

        avail_ratios = {k: [] for k in config.RESOURCE_TYPE}
        load_ratios  = {k: [] for k in config.RESOURCE_TYPE}
        mm1_vals, link_loads = [], []

        for node in dc_nodes:
            res = node.get_min_available_resource(t_start, t_end)
            cap = node.cap
            for k in config.RESOURCE_TYPE:
                avail_ratios[k].append(res[k] / max(cap[k], 1e-6))
                load_ratios[k].append((cap[k] - res[k]) / max(cap[k], 1e-6))
            loads    = [min((cap[k] - res[k]) / max(cap[k], 1e-6), 0.999)
                        for k in config.RESOURCE_TYPE]
            avg_load = sum(loads) / len(loads)
            omega    = max(1.0 - avg_load, 0.001)
            mm1_vals.append(min(avg_load / omega / 20.0, 1.0))

        for lnk in env.network.links:
            avail = lnk.get_available_bandwidth(t_start, t_end)
            link_loads.append((lnk.cap - avail) / max(lnk.cap, 1e-6))

        feats = (
            [float(np.mean(avail_ratios[k])) for k in config.RESOURCE_TYPE] +
            [float(np.mean(load_ratios[k]))  for k in config.RESOURCE_TYPE] +
            [float(np.mean(mm1_vals))] +
            [float(np.mean(link_loads)) if link_loads else 0.0]
        )
        return np.array(feats, np.float32)

    @staticmethod
    def extract_gq_features(sfc) -> np.ndarray:
        """5-dim: đặc trưng của request"""
        req = sfc.request
        if not req.vnfs:
            return np.zeros(GQ_DIM, np.float32)
        mean_res = {k: float(np.mean([v.resource.get(k, 0.0) for v in req.vnfs]))
                    for k in config.RESOURCE_TYPE}
        return np.array(
            [mean_res[k] for k in config.RESOURCE_TYPE] +
            [req.bw, len(req.vnfs) / 10.0],
            np.float32)

    @staticmethod
    def extract_oq_features(plan: dict, env, sfc) -> np.ndarray:
        """
        3-dim: chất lượng của placement plan (Oq trong EAC).
        Revenue, Cost, R2C — normalized về [0,1].
        """
        req      = sfc.request
        duration = max(req.end_time - req.arrival_time, 1e-6)

        rev_node = sum(sum(v.resource.values()) for v in req.vnfs)
        rev_link = req.bw * len(plan.get("links", {}))
        revenue  = duration * (rev_node + rev_link)

        cost = 0.0
        for v in plan.get("nodes", {}).values():
            dc_name  = v["dc"]
            vnf_name = v.get("vnf_name", "")
            if dc_name in env.network.nodes and vnf_name in env.vnfs:
                node = env.network.nodes[dc_name]
                vnf  = env.vnfs[vnf_name]
                c    = node.get_cost(vnf)
                if c < float('inf'):
                    cost += c
        for lp in plan.get("links", {}).values():
            cost += req.bw * max(len(lp.get("path", [])) - 1, 0)

        cost    = max(cost, 1e-6)
        r2c     = revenue / cost

        rev_norm  = min(revenue / 1000.0, 1.0)
        cost_norm = min(cost    / 1000.0, 1.0)
        r2c_norm  = min(r2c     / 2.0,    1.0)

        return np.array([rev_norm, cost_norm, r2c_norm], np.float32)

    # ── History & window ──────────────────────────────────────

    def push_triplet(self, gp: np.ndarray, gq: np.ndarray, oq: np.ndarray):
        self._history.append(np.concatenate([gp, gq, oq]).astype(np.float32))

    def _get_window(self) -> np.ndarray:
        """(1, window_size, triplet_dim) — pad 0 nếu chưa đủ lịch sử"""
        window = list(self._history)
        while len(window) < self.window_size:
            window.insert(0, np.zeros(self.triplet_dim, np.float32))
        arr = np.stack(window[-self.window_size:], axis=0)
        return arr[np.newaxis]                  # (1, W, D)

    def reset_history(self):
        """Gọi đầu mỗi episode để xóa lịch sử"""
        self._history.clear()

    # ── Inference ─────────────────────────────────────────────

    def decide(self, gp: np.ndarray, gq: np.ndarray, oq: np.ndarray,
               training: bool = False) -> tuple:
        """
        Returns: (accept: bool, log_prob: float, value: float)
        training=True  → sample từ distribution (explore)
        training=False → argmax (exploit)
        """
        self.push_triplet(gp, gq, oq)
        window = tf.constant(self._get_window(), dtype=tf.float32)

        logits = self.actor(window,  training=False).numpy()[0]
        value  = self.critic(window, training=False).numpy()[0, 0]

        probs  = tf.nn.softmax(logits).numpy()
        if training:
            action = int(np.random.choice(2, p=probs))
        else:
            action = int(np.argmax(probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        return bool(action == 1), log_prob, float(value)

    # ── PPO trajectory ────────────────────────────────────────

    def record(self, window: np.ndarray, action: int,
               log_prob: float, value: float, reward: float):
        self._traj.append((window, action, log_prob, value, reward))

    def clear_traj(self):
        self._traj.clear()

    # ── PPO update (gọi cuối mỗi episode) ────────────────────

    def train_ppo(self, gamma: float = 0.99, lam: float = 0.95,
                  clip_eps: float = 0.2, epochs: int = 4,
                  d1: float = 0.5, d2: float = 0.01) -> float | None:
        """GAE advantages + PPO clipped objective"""
        if len(self._traj) < 2:
            self.clear_traj()
            return None

        windows, actions, old_lps, values, rewards = zip(*self._traj)
        windows = np.concatenate(windows, axis=0)   # (T, W, D)
        actions = np.array(actions,  np.int32)
        old_lps = np.array(old_lps,  np.float32)
        values  = np.array(values,   np.float32)
        rewards = np.array(rewards,  np.float32)

        # GAE
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val    = values[t + 1] if t + 1 < len(values) else 0.0
            delta       = rewards[t] + gamma * next_val - values[t]
            gae         = delta + gamma * lam * gae
            advantages[t] = gae
        returns    = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        W_t   = tf.constant(windows,    tf.float32)
        A_t   = tf.constant(advantages, tf.float32)
        R_t   = tf.constant(returns,    tf.float32)
        act_t = tf.constant(actions,    tf.int32)
        olp_t = tf.constant(old_lps,    tf.float32)

        total_loss = 0.0
        for _ in range(epochs):
            with tf.GradientTape() as tape_a:
                logits    = self.actor(W_t, training=True)
                log_probs = tf.nn.log_softmax(logits)
                idx       = tf.stack([tf.range(tf.shape(act_t)[0]), act_t], axis=1)
                new_lp    = tf.gather_nd(log_probs, idx)
                ratio     = tf.exp(new_lp - olp_t)
                entropy   = -tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.softmax(logits) * tf.nn.log_softmax(logits),
                        axis=-1))
                surr1      = ratio * A_t
                surr2      = tf.clip_by_value(ratio, 1 - clip_eps, 1 + clip_eps) * A_t
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - d2 * entropy

            self.opt_actor.apply_gradients(
                zip(tape_a.gradient(actor_loss, self.actor.trainable_variables),
                    self.actor.trainable_variables))

            with tf.GradientTape() as tape_c:
                v_pred      = self.critic(W_t, training=True)[:, 0]
                critic_loss = d1 * tf.reduce_mean(tf.square(R_t - v_pred))

            self.opt_critic.apply_gradients(
                zip(tape_c.gradient(critic_loss, self.critic.trainable_variables),
                    self.critic.trainable_variables))

            total_loss += float(actor_loss + critic_loss)

        self.clear_traj()
        return total_loss / epochs

    # ── Persistence ───────────────────────────────────────────

    def save_weights(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.actor.save_weights(
            os.path.join(directory, "admission_actor.weights.h5"))
        self.critic.save_weights(
            os.path.join(directory, "admission_critic.weights.h5"))
        print(f"[Admission] Saved -> {directory}")

    def load_weights(self, directory: str):
        ap = os.path.join(directory, "admission_actor.weights.h5")
        cp = os.path.join(directory, "admission_critic.weights.h5")
        if os.path.exists(ap):
            self.actor.load_weights(ap)
        if os.path.exists(cp):
            self.critic.load_weights(cp)
        if os.path.exists(ap) or os.path.exists(cp):
            print(f"[Admission] Loaded <- {directory}")