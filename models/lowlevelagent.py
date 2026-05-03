from __future__ import annotations

import os, random, collections
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from models.model import _mlp, ReplayBuffer

class LowLevelAgent:
    def __init__(self, latent_dim: int = 8, max_dcs: int = 50,
                 gamma: float = 0.95, lr: float = 5e-4,
                 input_dim: Optional[int] = None):
        self.latent_dim = latent_dim
        self.max_dcs    = max_dcs
        self.gamma      = gamma

        feat_dim = input_dim if input_dim else latent_dim * 2 + 3
        self.policy_net = _mlp(feat_dim, 128, max_dcs, "ll_policy")
        self.target_net = _mlp(feat_dim, 128, max_dcs, "ll_target")
        self._sync_target()
        self.opt = keras.optimizers.Adam(lr)
        self.weight_net = _mlp(feat_dim, 32, 2, "ll_weights")
        self.opt_w      = keras.optimizers.Adam(lr)

        self._reward_mean = 0.0
        self._reward_var  = 1.0
        self._reward_count = 0

    def _sync_target(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def update_target_network(self):
        self._sync_target()

    def _safe_z_mean(self, Z_t: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(Z_t, dtype=np.float32)
        if z_arr.size == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        if z_arr.ndim == 2:
            if z_arr.shape[0] == 0:
                return np.zeros(self.latent_dim, dtype=np.float32)
            return z_arr.mean(axis=0)
        flat = z_arr.ravel()
        if flat.size == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        if flat.size < self.latent_dim:
            out = np.zeros(self.latent_dim, dtype=np.float32)
            out[:flat.size] = flat
            return out
        return flat[:self.latent_dim]

    def _make_state(self, Z_t: np.ndarray, vnf_feat: list,
                    loc_z: Optional[np.ndarray] = None) -> np.ndarray:
        z_mean = self._safe_z_mean(Z_t)
        f_arr  = np.asarray(vnf_feat, dtype=np.float32).ravel()
        if loc_z is None:
            loc_z = np.zeros(self.latent_dim, dtype=np.float32)
        loc_arr = np.asarray(loc_z, dtype=np.float32).ravel()[:self.latent_dim]
        if loc_arr.shape[0] < self.latent_dim:
            pad = np.zeros(self.latent_dim, dtype=np.float32)
            pad[:loc_arr.shape[0]] = loc_arr
            loc_arr = pad
        return np.concatenate([z_mean, f_arr, loc_arr])[None]

    def _make_states_batch(self, Z_list: list, vnf_feat_list: list,
                           loc_list: Optional[list] = None) -> np.ndarray:
        rows = []
        for i, (Z, f) in enumerate(zip(Z_list, vnf_feat_list)):
            z_mean = self._safe_z_mean(Z)
            loc_z  = (np.asarray(loc_list[i], np.float32).ravel()
                      if loc_list and loc_list[i] is not None
                      else np.zeros(self.latent_dim, np.float32))
            if loc_z.shape[0] < self.latent_dim:
                pad = np.zeros(self.latent_dim, np.float32)
                pad[:loc_z.shape[0]] = loc_z
                loc_z = pad
            else:
                loc_z = loc_z[:self.latent_dim]
            rows.append(np.concatenate([z_mean, np.asarray(f, np.float32).ravel(), loc_z]))
        return np.array(rows, dtype=np.float32)
    
    def act(self, Z_t: np.ndarray, vnf_feat: list,
            valid_indices: List[int], epsilon: float = 0.0,
            loc_z: Optional[np.ndarray] = None) -> int:
        if not valid_indices:
            return 0
        if random.random() < epsilon:
            return random.choice(valid_indices)
        state = self._make_state(Z_t, vnf_feat, loc_z)
        q     = self.policy_net(state, training=False).numpy()[0]
        mask             = np.full(self.max_dcs, -1e9, dtype=np.float32)
        valid_clip       = [i for i in valid_indices if i < self.max_dcs]
        if not valid_clip:
            return valid_indices[0]
        mask[valid_clip] = q[valid_clip]
        return int(np.argmax(mask))

    def get_reward_weights(self, Z_t: np.ndarray, vnf_feat: list,
                           loc_z: Optional[np.ndarray] = None) -> Tuple[float, float]:
        state = self._make_state(Z_t, vnf_feat, loc_z)
        w     = tf.sigmoid(self.weight_net(state, training=False)).numpy()[0]
        return float(w[0]) * 2.0, float(w[1]) * 1.0

    @tf.function
    def _tf_train_step(self, S, R, D, A, Q_next_max):
        target = R + self.gamma * Q_next_max * (1.0 - D)
        idx = tf.stack([tf.range(tf.shape(A)[0]), A], axis=1)
        
        with tf.GradientTape() as tape:
            Q_pred = self.policy_net(S, training=True)
            loss   = tf.reduce_mean(tf.square(target - tf.gather_nd(Q_pred, idx)))
        self.opt.apply_gradients(
            zip(tape.gradient(loss, self.policy_net.trainable_variables),
                self.policy_net.trainable_variables))

        R_pos = tf.maximum(tf.expand_dims(R, 1), 0.0)
        with tf.GradientTape() as tape_w:
            w_pred = tf.sigmoid(self.weight_net(S, training=True))
            loss_w = tf.reduce_mean(
                tf.square(w_pred - R_pos / (tf.reduce_max(R_pos) + 1e-6)))
        self.opt_w.apply_gradients(
            zip(tape_w.gradient(loss_w, self.weight_net.trainable_variables),
                self.weight_net.trainable_variables))

    def train(self, buffer: ReplayBuffer, batch_size: int = 16):
        if len(buffer) < batch_size:
            return
        batch = buffer.sample(batch_size)
        Z_list, vnf_f_list, loc_list, actions, rewards, Z_next_list, next_masks, loc_next_list, dones = zip(*batch)

        raw_rewards = np.array(
            [float(r[0]) if hasattr(r, '__len__') else float(r) for r in rewards],
            dtype=np.float32,
        )

        for r in raw_rewards:
            self._reward_count += 1
            delta = r - self._reward_mean
            self._reward_mean += delta / self._reward_count
            self._reward_var  += delta * (r - self._reward_mean)

        std = max(np.sqrt(self._reward_var / max(self._reward_count, 1)), 1e-6)
        normalized_rewards = (raw_rewards - self._reward_mean) / std

        S  = tf.constant(self._make_states_batch(Z_list, vnf_f_list, loc_list),       dtype=tf.float32)
        Sn = tf.constant(self._make_states_batch(Z_next_list, vnf_f_list, loc_next_list), dtype=tf.float32)
        R  = tf.constant(normalized_rewards, dtype=tf.float32)
        D  = tf.constant(np.array(dones, dtype=np.float32))
        A  = np.array([int(a) for a in actions], dtype=np.int32)

        Q_next_np = self.target_net(Sn, training=False).numpy()
        for i, mask in enumerate(next_masks):
            valid_clip = [m for m in mask if isinstance(m, int) and m < self.max_dcs]
            row = np.full(self.max_dcs, -1e9, dtype=np.float32)
            if valid_clip:
                row[valid_clip] = Q_next_np[i, valid_clip]
            Q_next_np[i] = row

        Q_next_max = tf.constant(Q_next_np.max(axis=1), dtype=tf.float32)
        self._tf_train_step(S, R, D, tf.constant(A, dtype=tf.int32), Q_next_max)
