from __future__ import annotations

import os, random, collections
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from models.model import _mlp, ReplayBuffer
from models.lowlevelagent import LowLevelAgent

class HighLevelAgent:
    FEAT_PER_SFC = 4

    def __init__(self, latent_dim: int = 8, max_queue: int = 20,
                 gamma: float = 0.95, lr: float = 5e-4,
                 use_ll_score: bool = True,
                 input_dim: Optional[int] = None):
        self.latent_dim   = latent_dim
        self.max_queue    = max_queue
        self.gamma        = gamma
        self.use_ll_score = use_ll_score
        feat_dim = input_dim if input_dim else latent_dim + self.FEAT_PER_SFC
        self.policy_net_ar   = _mlp(feat_dim, 128, 1, "hl_policy_ar")
        self.policy_net_cost = _mlp(feat_dim, 128, 1, "hl_policy_cost")
        self.target_net_ar   = _mlp(feat_dim, 128, 1, "hl_target_ar")
        self.target_net_cost = _mlp(feat_dim, 128, 1, "hl_target_cost")
        self.opt_ar   = keras.optimizers.Adam(lr)
        self.opt_cost = keras.optimizers.Adam(lr)
        self.log_std  = tf.Variable(tf.zeros(2), trainable=True, name="hl_log_std")
        self._sync_targets()

    def _sync_targets(self):
        self.target_net_ar.set_weights(self.policy_net_ar.get_weights())
        self.target_net_cost.set_weights(self.policy_net_cost.get_weights())

    def update_target_networks(self):
        self._sync_targets()

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

    def extract_sfc_features(self, queue: list, Z_t: Optional[np.ndarray] = None,
                          ll_agent: Optional[LowLevelAgent] = None) -> np.ndarray:
        if not queue:
            return np.zeros((0, self.FEAT_PER_SFC), dtype=np.float32)

        max_delay = max((s.request.delay_max for s in queue), default=1.0)
        max_delay = max(max_delay, 1.0)

        ll_scores = np.full(len(queue), 0.5, dtype=np.float32)
        if self.use_ll_score and Z_t is not None and ll_agent is not None:
            states  = []
            has_vnf = []
            for sfc in queue:
                if sfc.request.vnfs:
                    vnf_feat = [sfc.request.vnfs[0].resource.get(k, 0)
                                for k in ["mem", "cpu", "ram"]]
                    states.append(ll_agent._make_state(Z_t, vnf_feat)[0])
                    has_vnf.append(True)
                else:
                    has_vnf.append(False)
            if states:
                S_batch = tf.constant(np.array(states, dtype=np.float32))
                Q_batch = ll_agent.policy_net(S_batch, training=False).numpy()
                j = 0
                for i, has in enumerate(has_vnf):
                    if has:
                        q_max = float(Q_batch[j].max())
                        ll_scores[i] = q_max / (abs(q_max) + 1.0)
                        j += 1

        feats = np.array([
            [sfc.request.bw,
            float(len(sfc.request.vnfs)),
            min(1.0, sfc.request.delay_max / max_delay),
            float(ll_scores[i])]
            for i, sfc in enumerate(queue)
        ], dtype=np.float32)
        return feats
    
    def get_weights(self) -> dict:
        return {
            'ar':   self.policy_net_ar.get_weights(),
            'cost': self.policy_net_cost.get_weights(),
        }

    def set_weights(self, weights: dict):
        if 'ar' in weights:
            self.policy_net_ar.set_weights(weights['ar'])
            self.target_net_ar.set_weights(weights['ar'])
        if 'cost' in weights:
            self.policy_net_cost.set_weights(weights['cost'])
            self.target_net_cost.set_weights(weights['cost'])

    def _state_for(self, Z_t: np.ndarray, sfc_feat) -> np.ndarray:
        z_mean = self._safe_z_mean(Z_t)
        return np.concatenate([z_mean, np.asarray(sfc_feat, np.float32).ravel()])[None]

    def _states_batch(self, Z_mean: np.ndarray, sfc_feats: np.ndarray) -> np.ndarray:
        z = np.broadcast_to(Z_mean[None], (len(sfc_feats), len(Z_mean)))
        return np.concatenate([z, sfc_feats], axis=1).astype(np.float32)

    @staticmethod
    def _nondominated_sort(q_matrix: np.ndarray) -> List[int]:
        """
        q_matrix[:, 0] = Q_acceptance (higher better)
        q_matrix[:, 1] = Q_cost (higher = less cost = better, because R_cost = -cost_norm)
        Both objectives: higher is better. No negation needed.
        """
        N         = len(q_matrix)
        dom_count = np.zeros(N, dtype=int)
        for i in range(N):
            for j in range(N):
                if i != j:
                    if (q_matrix[j, 0] >= q_matrix[i, 0] and
                            q_matrix[j, 1] >= q_matrix[i, 1] and
                            (q_matrix[j, 0] > q_matrix[i, 0] or
                             q_matrix[j, 1] > q_matrix[i, 1])):
                        dom_count[i] += 1
        front = list(np.where(dom_count == 0)[0])
        return front if front else list(range(N))
    
    def _crowding_distance_selection(self, front_indices: List[int], q_mat: np.ndarray) -> int:
        if len(front_indices) <= 2:
            return random.choice(front_indices)

        n = len(front_indices)
        distances = np.zeros(n)
        
        for obj_idx in [0, 1]:
            sorted_keys = sorted(range(n), key=lambda k: q_mat[front_indices[k], obj_idx])
            
            distances[sorted_keys[0]] = 1e9
            distances[sorted_keys[-1]] = 1e9
            
            obj_min = q_mat[front_indices[sorted_keys[0]], obj_idx]
            obj_max = q_mat[front_indices[sorted_keys[-1]], obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_keys[i]] += (q_mat[front_indices[sorted_keys[i+1]], obj_idx] - 
                                                q_mat[front_indices[sorted_keys[i-1]], obj_idx]) / obj_range
        
        return front_indices[np.argmax(distances)]

    def act(self, Z_t: np.ndarray, queue: list, epsilon: float = 0.0,
            ll_agent: Optional[LowLevelAgent] = None) -> int:
        if not queue:
            return 0
        if len(queue) == 1:
            return 0
        if random.random() < epsilon:
            return random.randrange(len(queue))

        sfc_feats = self.extract_sfc_features(queue, Z_t, ll_agent)
        z_mean    = self._safe_z_mean(Z_t)
        S_batch = tf.constant(self._states_batch(z_mean, sfc_feats))
        
        q_ar   = self.policy_net_ar(S_batch,   training=False).numpy()
        q_cost = self.policy_net_cost(S_batch, training=False).numpy()
        q_mat  = np.concatenate([q_ar, q_cost], axis=1)

        front = self._nondominated_sort(q_mat)
        return self._crowding_distance_selection(front, q_mat)

    def train(self, buffer: ReplayBuffer, batch_size: int = 16):
        if len(buffer) < batch_size:
            return
        batch = buffer.sample(batch_size)

        states, actions, rewards_ar, rewards_cost, dones, states_next = [], [], [], [], [], []

        for (Z_mean, sfc_feats, sfc_idx, R_HL, Z_next_mean, sfc_feats_next, done) in batch:
            if len(sfc_feats) == 0:
                continue
            idx_clip = min(int(sfc_idx), len(sfc_feats) - 1)
            if idx_clip < 0:
                continue

            z   = np.asarray(Z_mean, np.float32).ravel()
            f   = np.asarray(sfc_feats[idx_clip], np.float32).ravel()
            states.append(np.concatenate([z, f]))

            z_next = np.asarray(Z_next_mean, np.float32).ravel()
            if len(sfc_feats_next) > 0:
                f_next = np.asarray(sfc_feats_next[0], np.float32).ravel()
            else:
                f_next = np.zeros_like(f)
            states_next.append(np.concatenate([z_next, f_next]))

            actions.append(idx_clip)
            r_ar   = float(R_HL[0]) if hasattr(R_HL, '__len__') else float(R_HL)
            r_cost = float(R_HL[1]) if hasattr(R_HL, '__len__') and len(R_HL) > 1 else 0.0
            rewards_ar.append(r_ar)
            rewards_cost.append(r_cost)
            dones.append(float(done))

        if not states:
            return

        S      = tf.constant(np.array(states,      np.float32))
        S_next = tf.constant(np.array(states_next, np.float32))
        R_ar   = tf.constant(np.array(rewards_ar,   np.float32))
        R_cost = tf.constant(np.array(rewards_cost, np.float32))
        D      = tf.constant(np.array(dones,        np.float32))

        Q_next_ar   = tf.stop_gradient(tf.squeeze(self.target_net_ar(S_next,   training=False), axis=1))
        Q_next_cost = tf.stop_gradient(tf.squeeze(self.target_net_cost(S_next, training=False), axis=1))

        target_ar   = R_ar   + self.gamma * Q_next_ar   * (1.0 - D)
        target_cost = R_cost + self.gamma * Q_next_cost * (1.0 - D)

        with tf.GradientTape() as tape_ar:
            Q_pred_ar = tf.squeeze(self.policy_net_ar(S, training=True), axis=1)
            loss_ar   = tf.reduce_mean(tf.square(target_ar - Q_pred_ar))
        self.opt_ar.apply_gradients(
            zip(tape_ar.gradient(loss_ar, self.policy_net_ar.trainable_variables),
                self.policy_net_ar.trainable_variables)
        )

        with tf.GradientTape() as tape_cost:
            Q_pred_cost = tf.squeeze(self.policy_net_cost(S, training=True), axis=1)
            loss_cost   = tf.reduce_mean(tf.square(target_cost - Q_pred_cost))
        self.opt_cost.apply_gradients(
            zip(tape_cost.gradient(loss_cost, self.policy_net_cost.trainable_variables),
                self.policy_net_cost.trainable_variables)
        )