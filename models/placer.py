from __future__ import annotations
import math, random, collections
from typing import List, Optional, Tuple
import numpy as np

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from models.model import _mlp, ReplayBuffer
import config


class PressureNode:
    @staticmethod
    def compute(remaining: dict, demand: dict, capacity: dict) -> float:
        pressures = []
        for k in config.RESOURCE_TYPE:
            cap  = capacity.get(k, 1.0)
            used = cap - remaining.get(k, 0.0)
            load = min(used / max(cap, 1e-6), 0.999)
            omega = 1.0 - load
            pressures.append(min(load / max(omega, 0.001) / 20.0, 1.0))
        return sum(pressures) / len(pressures) if pressures else 0.0


class PlacerAgent:
    FEAT_DIM_EXTRA = 3 + 1
    _REWARD_WINDOW = 200

    def __init__(self, latent_dim: int = None, max_dcs: int = None,
                 gamma: float = 0.95, lr: float = None,
                 input_dim: Optional[int] = None):
        import config as cfg
        self.latent_dim = latent_dim or cfg.LATENT_DIM
        self.max_dcs    = max_dcs    or cfg.MAX_DCS
        self.gamma      = gamma
        self.feat_dim   = (input_dim if input_dim
                           else self.latent_dim * 2 + self.FEAT_DIM_EXTRA)
        hidden          = getattr(cfg, 'PLACER_HIDDEN_DIM', 256)
        _lr             = lr or getattr(cfg, 'PLACER_LR', 3e-4)

        self.policy_net = _mlp(self.feat_dim, hidden, self.max_dcs, "placer_policy")
        self.target_net = _mlp(self.feat_dim, hidden, self.max_dcs, "placer_target")
        self._sync_target()
        self.opt        = keras.optimizers.Adam(_lr)
        self.weight_net = _mlp(self.feat_dim, 64, 2, "placer_weights")
        self.opt_w      = keras.optimizers.Adam(_lr)
        self._reward_window = collections.deque(maxlen=self._REWARD_WINDOW)

    def _sync_target(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def update_target_network(self):
        self._sync_target()

    def _safe_z_mean(self, Z_t: np.ndarray) -> np.ndarray:
        z = np.asarray(Z_t, dtype=np.float32)
        if z.size == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        if z.ndim == 2:
            return z.mean(axis=0) if z.shape[0] > 0 else np.zeros(self.latent_dim, np.float32)
        flat = z.ravel()
        if flat.size < self.latent_dim:
            out = np.zeros(self.latent_dim, dtype=np.float32)
            out[:flat.size] = flat
            return out
        return flat[:self.latent_dim]

    def make_state(self, Z_t: np.ndarray, vnf_feat: list,
                   loc_z: Optional[np.ndarray] = None,
                   node_pressure: float = 0.0) -> np.ndarray:
        global_z = self._safe_z_mean(Z_t)
        f_arr    = np.asarray(vnf_feat, dtype=np.float32).ravel()
        if loc_z is None:
            loc_z = np.zeros(self.latent_dim, dtype=np.float32)
        loc_arr = np.asarray(loc_z, dtype=np.float32).ravel()[:self.latent_dim]
        if loc_arr.shape[0] < self.latent_dim:
            pad = np.zeros(self.latent_dim, dtype=np.float32)
            pad[:loc_arr.shape[0]] = loc_arr
            loc_arr = pad
        pressure_feat = np.array([node_pressure], dtype=np.float32)
        return np.concatenate([global_z, f_arr, loc_arr, pressure_feat])[None]

    def _make_states_batch(self, Z_list, vnf_feat_list,
                           loc_list=None, pressure_list=None) -> np.ndarray:
        rows = []
        for i, (Z, f) in enumerate(zip(Z_list, vnf_feat_list)):
            loc_z = (np.asarray(loc_list[i], np.float32).ravel()
                     if loc_list and loc_list[i] is not None
                     else np.zeros(self.latent_dim, np.float32))
            if loc_z.shape[0] < self.latent_dim:
                pad = np.zeros(self.latent_dim, np.float32)
                pad[:loc_z.shape[0]] = loc_z
                loc_z = pad
            else:
                loc_z = loc_z[:self.latent_dim]
            pressure = np.array([pressure_list[i] if pressure_list else 0.0], np.float32)
            rows.append(np.concatenate([self._safe_z_mean(Z),
                                        np.asarray(f, np.float32).ravel(),
                                        loc_z, pressure]))
        return np.array(rows, dtype=np.float32)

    def act(self, Z_t: np.ndarray, vnf_feat: list,
            valid_indices: List[int], epsilon: float = 0.0,
            loc_z: Optional[np.ndarray] = None,
            node_pressure: float = 0.0) -> int:
        if not valid_indices:
            return 0
        if random.random() < epsilon:
            return random.choice(valid_indices)
        state = self.make_state(Z_t, vnf_feat, loc_z, node_pressure)
        q     = self.policy_net(state, training=False).numpy()[0]
        mask  = np.full(self.max_dcs, -1e9, dtype=np.float32)
        valid_clip = [i for i in valid_indices if i < self.max_dcs]
        if not valid_clip:
            return valid_indices[0]
        mask[valid_clip] = q[valid_clip]
        return int(np.argmax(mask))

    def get_reward_weights(self, Z_t: np.ndarray, vnf_feat: list,
                           loc_z: Optional[np.ndarray] = None,
                           node_pressure: float = 0.0) -> Tuple[float, float]:
        state = self.make_state(Z_t, vnf_feat, loc_z, node_pressure)
        w     = tf.sigmoid(self.weight_net(state, training=False)).numpy()[0]
        return float(w[0]) * 2.0, float(w[1]) * 1.0

    @tf.function
    def _train_step(self, S, R, D, A, Q_next_max):
        target = R + self.gamma * Q_next_max * (1.0 - D)
        idx    = tf.stack([tf.range(tf.shape(A)[0]), A], axis=1)
        with tf.GradientTape() as tape:
            q_vals = tf.gather_nd(self.policy_net(S, training=True), idx)
            loss   = tf.reduce_mean(tf.square(target - q_vals))
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        R_pos = tf.maximum(tf.expand_dims(R, 1), 0.0)
        with tf.GradientTape() as tape_w:
            r_max  = tf.reduce_max(R_pos) + 1e-6
            loss_w = tf.reduce_mean(tf.square(
                tf.sigmoid(self.weight_net(S, training=True)) -
                R_pos / r_max))
        grads_w = tape_w.gradient(loss_w, self.weight_net.trainable_variables)
        grads_w = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads_w]
        self.opt_w.apply_gradients(zip(grads_w, self.weight_net.trainable_variables))

    def train(self, buffer: ReplayBuffer, batch_size: int = 16):
        if len(buffer) < batch_size:
            return
        batch = buffer.sample(batch_size)
        Z_list, vnf_f, loc_list, pressure_list, actions, rewards, \
            Z_next, next_masks, loc_next, pressure_next, dones = zip(*batch)

        raw = np.array(
            [float(r[0]) if hasattr(r, '__len__') else float(r) for r in rewards],
            dtype=np.float32)
        self._reward_window.extend(raw.tolist())
        w_mean = float(np.mean(self._reward_window))
        w_std  = max(float(np.std(self._reward_window)), 1e-6)
        norm_r = (raw - w_mean) / w_std

        S  = tf.constant(self._make_states_batch(Z_list, vnf_f, loc_list, pressure_list),
                         dtype=tf.float32)
        Sn = tf.constant(self._make_states_batch(Z_next, vnf_f, loc_next, pressure_next),
                         dtype=tf.float32)
        Q_next = self.target_net(Sn, training=False).numpy()
        for i, mask in enumerate(next_masks):
            valid_clip = [m for m in mask if isinstance(m, int) and m < self.max_dcs]
            row = np.full(self.max_dcs, -1e9, dtype=np.float32)
            if valid_clip:
                row[valid_clip] = Q_next[i, valid_clip]
            Q_next[i] = row

        self._train_step(
            S,
            tf.constant(norm_r, dtype=tf.float32),
            tf.constant(np.array(dones, dtype=np.float32)),
            tf.constant(np.array([int(a) for a in actions], dtype=np.int32)),
            tf.constant(Q_next.max(axis=1), dtype=tf.float32))