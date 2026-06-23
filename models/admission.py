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

GP_DIM      = 8
GQ_DIM      = 6
OQ_DIM      = 4
TRIPLET_DIM = GP_DIM + GQ_DIM + OQ_DIM
WINDOW_SIZE = config.ADMISSION_WINDOW_SIZE


class AdmissionAgent:
    def __init__(self, triplet_dim: int = TRIPLET_DIM,
                 window_size: int = WINDOW_SIZE,
                 hidden_dim: int = None,
                 lr: float = None):
        self.triplet_dim = triplet_dim
        self.window_size = window_size
        self.hidden_dim  = hidden_dim or config.ADMISSION_HIDDEN_DIM
        self._lr         = lr or config.ADMISSION_LR

        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._traj:    list              = []

        # Track whether weights have been trained (not just loaded randomly).
        self._is_trained: bool = False

        self.actor  = self._build_rnn_model("admission_actor",  out_dim=2)
        self.critic = self._build_rnn_model("admission_critic", out_dim=1)

        self.opt_actor  = keras.optimizers.Adam(self._lr)
        self.opt_critic = keras.optimizers.Adam(self._lr)

        _dummy = np.zeros((1, window_size, triplet_dim), np.float32)
        self.actor(_dummy)
        self.critic(_dummy)

    def _build_rnn_model(self, name: str, out_dim: int) -> keras.Model:
        inp = keras.Input(shape=(self.window_size, self.triplet_dim), name=name + "_in")
        x   = layers.GRU(self.hidden_dim, return_sequences=True)(inp)
        x   = layers.GRU(self.hidden_dim // 2, return_sequences=False)(x)
        x   = layers.Dense(self.hidden_dim // 2, activation="relu")(x)
        x   = layers.Dropout(0.1)(x)
        out = layers.Dense(out_dim)(x)
        return keras.Model(inp, out, name=name)

    # ── Feature extraction ────────────────────────────────────

    @staticmethod
    def extract_gp_features(env, t_start: int, t_end: int) -> np.ndarray:
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
        req = sfc.request
        if not req.vnfs:
            return np.zeros(GQ_DIM, np.float32)
        mean_res = {k: float(np.mean([v.resource.get(k, 0.0) for v in req.vnfs]))
                    for k in config.RESOURCE_TYPE}
        num_vnfs_norm = len(req.vnfs) / 10.0
        bw_norm       = req.bw / 100.0
        deadline_norm = min(1.0, (req.end_time - req.arrival_time) / 1000.0)
        return np.array(
            [mean_res[k] for k in config.RESOURCE_TYPE] +
            [bw_norm, num_vnfs_norm, deadline_norm],
            np.float32)

    @staticmethod
    def extract_oq_features(plan: dict, env, sfc) -> np.ndarray:
        req      = sfc.request
        duration = max(req.end_time - req.arrival_time, 1e-6)

        rev_node = sum(
            config.REVENUE_WEIGHT_NODE * sum(v.resource.values())
            for v in req.vnfs)
        num_links = len(plan.get("links", {}))
        rev_link  = config.REVENUE_WEIGHT_LINK * req.bw * num_links
        revenue   = duration * (rev_node + rev_link)

        node_cost = 0.0
        for v in plan.get("nodes", {}).values():
            dc_name  = v.get("dc", "")
            vnf_name = v.get("vnf_name", "")
            if dc_name in env.network.nodes and vnf_name in env.vnfs:
                node = env.network.nodes[dc_name]
                vnf  = env.vnfs[vnf_name]
                c    = node.get_cost(vnf)
                if c < float('inf'):
                    node_cost += config.REVENUE_WEIGHT_NODE * c

        link_cost = 0.0
        for lp in plan.get("links", {}).values():
            path_len  = max(len(lp.get("path", [])) - 1, 0)
            link_cost += config.REVENUE_WEIGHT_LINK * req.bw * path_len

        cost  = max(node_cost + link_cost * duration, 1e-6)
        r2c   = revenue / cost

        norm_scale = max(revenue, cost, 1.0)
        rev_norm   = min(revenue / norm_scale, 1.0)
        cost_norm  = min(cost    / norm_scale, 1.0)
        r2c_norm   = min(r2c / 3.0, 1.0)
        path_eff   = min(1.0 / (link_cost / max(revenue, 1e-6) + 1e-6) / 10.0, 1.0)

        return np.array([rev_norm, cost_norm, r2c_norm, path_eff], np.float32)

    # ── History management ────────────────────────────────────

    def push_triplet(self, gp: np.ndarray, gq: np.ndarray, oq: np.ndarray):
        self._history.append(np.concatenate([gp, gq, oq]).astype(np.float32))

    def _get_window(self) -> np.ndarray:
        pad    = np.zeros(self.triplet_dim, np.float32)
        window = list(self._history)
        while len(window) < self.window_size:
            window.insert(0, pad)
        arr = np.stack(window[-self.window_size:], axis=0)
        return arr[np.newaxis]

    def reset_history(self):
        self._history.clear()

    # ── Decision ──────────────────────────────────────────────

    def decide(self, gp: np.ndarray, gq: np.ndarray, oq: np.ndarray,
               training: bool = False,
               force_accept: bool = False) -> tuple:
        """
        Returns (accept: bool, log_prob: float, value: float).

        force_accept=True is used during warmup phases so the agent still
        records state/value for learning but does not reject anything.

        Static R2C floor: if ADMISSION_R2C_FLOOR > 0 and the plan's R2C
        (oq[2] * 3.0) is below the floor, reject immediately without
        consuming a trajectory step (so the agent is not penalized for a
        filter that is purely rule-based).

        Confidence threshold: only reject when P(reject) >=
        ADMISSION_REJECT_THRESHOLD. This prevents an untrained GRU from
        discarding requests because its random logits happen to favour
        class-0 (reject).
        """
        self.push_triplet(gp, gq, oq)
        window = tf.constant(self._get_window(), dtype=tf.float32)

        logits = self.actor(window,  training=False).numpy()[0]
        value  = self.critic(window, training=False).numpy()[0, 0]

        probs  = tf.nn.softmax(logits).numpy()
        probs  = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()

        if training:
            action = int(np.random.choice(2, p=probs))
        else:
            action = int(np.argmax(probs))

        log_prob = float(np.log(probs[action]))

        # --- force accept override ---
        if force_accept:
            return True, float(np.log(probs[1])), float(value)

        # --- static R2C floor (rule-based, always active) ---
        r2c_floor = getattr(config, 'ADMISSION_R2C_FLOOR', 0.0)
        if r2c_floor > 0.0:
            r2c_actual = float(oq[2]) * 3.0
            if r2c_actual < r2c_floor:
                return False, float(np.log(probs[0])), float(value)

        # --- confidence threshold: default to accept if not confident enough ---
        reject_threshold = getattr(config, 'ADMISSION_REJECT_THRESHOLD', 0.65)
        if not self._is_trained:
            # Untrained agent: always accept (never reject)
            accept = True
        elif action == 0:
            # Agent wants to reject — only do so if confident
            accept = probs[0] < reject_threshold
        else:
            accept = True

        return accept, log_prob, float(value)

    # ── PPO trajectory ────────────────────────────────────────

    def record(self, window: np.ndarray, action: int,
               log_prob: float, value: float, reward: float):
        self._traj.append((window, action, log_prob, value, reward))

    def clear_traj(self):
        self._traj.clear()

    # ── PPO update ────────────────────────────────────────────

    def train_ppo(self,
                  gamma:    float = None,
                  lam:      float = None,
                  clip_eps: float = None,
                  epochs:   int   = None,
                  d1:       float = None,
                  d2:       float = None) -> float | None:
        gamma    = gamma    or config.ADMISSION_GAMMA
        lam      = lam      or config.ADMISSION_LAM
        clip_eps = clip_eps or config.ADMISSION_CLIP_EPS
        epochs   = epochs   or config.ADMISSION_PPO_EPOCHS
        d1       = d1       or config.ADMISSION_VALUE_COEF
        d2       = d2       or config.ADMISSION_ENTROPY_COEF

        if len(self._traj) < 2:
            self.clear_traj()
            return None

        windows, actions, old_lps, values, rewards = zip(*self._traj)
        windows = np.concatenate(windows, axis=0)
        actions = np.array(actions,  np.int32)
        old_lps = np.array(old_lps,  np.float32)
        values  = np.array(values,   np.float32)
        rewards = np.array(rewards,  np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val      = values[t + 1] if t + 1 < len(values) else 0.0
            delta         = rewards[t] + gamma * next_val - values[t]
            gae           = delta + gamma * lam * gae
            advantages[t] = gae
        returns    = advantages + values
        adv_std    = advantages.std()
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

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
                probs_all = tf.nn.softmax(logits)
                entropy   = -tf.reduce_mean(
                    tf.reduce_sum(probs_all * tf.nn.log_softmax(logits), axis=-1))
                surr1      = ratio * A_t
                surr2      = tf.clip_by_value(ratio, 1 - clip_eps, 1 + clip_eps) * A_t
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - d2 * entropy

            grads_a = tape_a.gradient(actor_loss, self.actor.trainable_variables)
            grads_a = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads_a]
            self.opt_actor.apply_gradients(
                zip(grads_a, self.actor.trainable_variables))

            with tf.GradientTape() as tape_c:
                v_pred      = self.critic(W_t, training=True)[:, 0]
                critic_loss = d1 * tf.reduce_mean(tf.square(R_t - v_pred))

            grads_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)
            grads_c = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads_c]
            self.opt_critic.apply_gradients(
                zip(grads_c, self.critic.trainable_variables))

            total_loss += float(actor_loss + critic_loss)

        self._is_trained = True
        self.clear_traj()
        return total_loss / epochs

    # ── Persistence ───────────────────────────────────────────

    def save_weights(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.actor.save_weights(
            os.path.join(directory, "admission_actor.weights.h5"))
        self.critic.save_weights(
            os.path.join(directory, "admission_critic.weights.h5"))

    def load_weights(self, directory: str):
        ap = os.path.join(directory, "admission_actor.weights.h5")
        cp = os.path.join(directory, "admission_critic.weights.h5")
        loaded = False
        if os.path.exists(ap):
            self.actor.load_weights(ap)
            loaded = True
        if os.path.exists(cp):
            self.critic.load_weights(cp)
            loaded = True
        if loaded:
            self._is_trained = True