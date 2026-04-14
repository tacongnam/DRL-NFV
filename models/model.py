from __future__ import annotations

import os, random, collections
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, transition):
        self.buf.append(transition)

    def sample(self, n: int):
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)

class GCNLayer(layers.Layer):
    def __init__(self, out_dim: int, activation="relu", **kw):
        super().__init__(**kw)
        self.dense = layers.Dense(out_dim, use_bias=False)
        self.act   = layers.Activation(activation) if activation else None

    def call(self, H, A_hat):
        out = self.dense(tf.matmul(A_hat, H))
        return self.act(out) if self.act else out

class VGAENetwork:
    def __init__(self, node_feat_dim: int = 3, hidden_dim: int = 16,
                 latent_dim: int = 8, lr: float = 1e-3, beta: float = 1e-3):
        self.latent_dim = latent_dim
        self.beta       = beta
        self._built     = False

        self.gcn1   = GCNLayer(hidden_dim, activation="relu",  name="gcn1")
        self.gcn_mu = GCNLayer(latent_dim, activation=None,    name="gcn_mu")
        self.gcn_lv = GCNLayer(latent_dim, activation=None,    name="gcn_logvar")
        self.optimizer = keras.optimizers.Adam(lr)

    @staticmethod
    def _norm_adj(A: np.ndarray) -> tf.Tensor:
        N   = A.shape[0]
        Ai  = A + np.eye(N, dtype=np.float32)
        deg = Ai.sum(axis=1, keepdims=True).clip(min=1)
        D   = np.diag(1.0 / np.sqrt(deg.flatten()))
        return tf.constant(D @ Ai @ D, dtype=tf.float32)

    def encode(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        X_t   = tf.constant(X, dtype=tf.float32)
        A_hat = self._norm_adj(A.astype(np.float32))
        mu    = self.gcn_mu(self.gcn1(X_t, A_hat), A_hat)
        self._built = True
        return mu.numpy()

    def _train_step(self, X_t: tf.Tensor, A_hat: tf.Tensor, A_t: tf.Tensor):
        with tf.GradientTape() as tape:
            h       = self.gcn1(X_t, A_hat)
            mu      = self.gcn_mu(h, A_hat)
            log_var = self.gcn_lv(h, A_hat)
            z       = mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))
            A_pred  = tf.sigmoid(tf.matmul(z, tf.transpose(z)))
            eps     = 1e-7
            bce     = -tf.reduce_mean(
                A_t * tf.math.log(A_pred + eps) +
                (1 - A_t) * tf.math.log(1 - A_pred + eps))
            kl   = -0.5 * tf.reduce_mean(1 + log_var - mu**2 - tf.exp(log_var))
            loss = bce + self.beta * kl
        vars_ = (self.gcn1.trainable_variables
                 + self.gcn_mu.trainable_variables
                 + self.gcn_lv.trainable_variables)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
        return loss.numpy()

    def train(self, buffer: ReplayBuffer, epochs: int = 1, batch: int = 16):
        if len(buffer) < 4:
            return
        for _ in range(epochs):
            for X, A in buffer.sample(batch):
                if X.shape[0] < 2:
                    continue
                A_hat = self._norm_adj(A.astype(np.float32))
                self._train_step(
                    tf.constant(X, dtype=tf.float32),
                    A_hat,
                    tf.constant(A, dtype=tf.float32))

    def save_weights(self, path: str):
        self._ensure_built()
        np.save(path, {
            "gcn1":   [v.numpy() for v in self.gcn1.trainable_variables],
            "gcn_mu": [v.numpy() for v in self.gcn_mu.trainable_variables],
            "gcn_lv": [v.numpy() for v in self.gcn_lv.trainable_variables],
        }, allow_pickle=True)

    def load_weights(self, path: str):
        try:
            w = np.load(path, allow_pickle=True).item()
            self._ensure_built()
            for layer, key in [(self.gcn1, "gcn1"), (self.gcn_mu, "gcn_mu"), (self.gcn_lv, "gcn_lv")]:
                if key in w:
                    for var, val in zip(layer.trainable_variables, w[key]):
                        var.assign(val)
        except Exception as e:
            print(f"[VGAE] Could not load weights: {e}")

    def _ensure_built(self):
        if not self._built:
            self.encode(np.zeros((2, 3), np.float32), np.eye(2, dtype=np.float32))

def _mlp(input_dim: int, hidden: int, out_dim: int, name: str) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name=name + "_in")
    x   = layers.Dense(hidden, activation="relu")(inp)
    x   = layers.Dense(hidden // 2, activation="relu")(x)
    out = layers.Dense(out_dim)(x)
    return keras.Model(inp, out, name=name)

class LowLevelAgent:
    def __init__(self, latent_dim: int = 8, max_dcs: int = 50,
                 gamma: float = 0.95, lr: float = 5e-4,
                 input_dim: Optional[int] = None):
        self.latent_dim = latent_dim
        self.max_dcs    = max_dcs
        self.gamma      = gamma

        feat_dim = input_dim if input_dim else latent_dim + 3
        self.policy_net = _mlp(feat_dim, 128, max_dcs, "ll_policy")
        self.target_net = _mlp(feat_dim, 128, max_dcs, "ll_target")
        self._sync_target()
        self.opt = keras.optimizers.Adam(lr)

        self.weight_net = _mlp(feat_dim, 32, 2, "ll_weights")
        self.opt_w      = keras.optimizers.Adam(lr)

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

    def _make_state(self, Z_t: np.ndarray, vnf_feat) -> np.ndarray:
        z_mean = self._safe_z_mean(Z_t)
        f_arr  = np.asarray(vnf_feat, dtype=np.float32).ravel()
        return np.concatenate([z_mean, f_arr])[None]

    def _make_states_batch(self, Z_list: list, vnf_feat_list: list) -> np.ndarray:
        rows = []
        for Z, f in zip(Z_list, vnf_feat_list):
            z_mean = self._safe_z_mean(Z)
            rows.append(np.concatenate([z_mean, np.asarray(f, np.float32).ravel()]))
        return np.array(rows, dtype=np.float32)

    def act(self, Z_t: np.ndarray, vnf_feat: list,
            valid_indices: List[int], epsilon: float = 0.0) -> int:
        if not valid_indices:
            return 0
        if random.random() < epsilon:
            return random.choice(valid_indices)
        state = self._make_state(Z_t, vnf_feat)
        q     = self.policy_net(state, training=False).numpy()[0]
        mask             = np.full(self.max_dcs, -1e9, dtype=np.float32)
        valid_clip       = [i for i in valid_indices if i < self.max_dcs]
        if not valid_clip:
            return valid_indices[0]
        mask[valid_clip] = q[valid_clip]
        return int(np.argmax(mask))

    def get_reward_weights(self, Z_t: np.ndarray, vnf_feat: list) -> Tuple[float, float]:
        state = self._make_state(Z_t, vnf_feat)
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

        Z_list, vnf_f_list, actions, rewards, Z_next_list, next_masks, dones = zip(*batch)
        S  = tf.constant(self._make_states_batch(Z_list, vnf_f_list), dtype=tf.float32)
        Sn = tf.constant(self._make_states_batch(Z_next_list, vnf_f_list), dtype=tf.float32)

        R = tf.constant(
            np.array([float(r[0]) if hasattr(r, '__len__') else float(r) for r in rewards],
                     dtype=np.float32))
        D = tf.constant(np.array(dones, dtype=np.float32))
        A = np.array([int(a) for a in actions], dtype=np.int32)

        Q_next_np = self.target_net(Sn, training=False).numpy()
        for i, mask in enumerate(next_masks):
            valid_clip = [m for m in mask if isinstance(m, int) and m < self.max_dcs]
            if valid_clip:
                row = np.full(self.max_dcs, -1e9, dtype=np.float32)
                row[valid_clip] = Q_next_np[i, valid_clip]
                Q_next_np[i]   = row
            else:
                Q_next_np[i]   = -1e9

        Q_next_max = tf.constant(Q_next_np.max(axis=1), dtype=tf.float32)
        
        self._tf_train_step(S, R, D, tf.constant(A, dtype=tf.int32), Q_next_max)

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
        self.policy_net = _mlp(feat_dim, 128, 2, "hl_policy")
        self.target_net = _mlp(feat_dim, 128, 2, "hl_target")
        self._sync_target()
        self.opt = keras.optimizers.Adam(lr)

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

    def extract_sfc_features(self, queue: list, Z_t: Optional[np.ndarray] = None,
                              ll_agent: Optional[LowLevelAgent] = None) -> np.ndarray:
        if not queue:
            return np.zeros((0, self.FEAT_PER_SFC), dtype=np.float32)

        max_delay = max((s.request.delay_max for s in queue), default=1.0)
        max_delay = max(max_delay, 1.0)

        ll_scores = np.full(len(queue), 0.5, dtype=np.float32)
        if self.use_ll_score and Z_t is not None and ll_agent is not None:
            states = []
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

    def _state_for(self, Z_t: np.ndarray, sfc_feat) -> np.ndarray:
        z_mean = self._safe_z_mean(Z_t)
        return np.concatenate([z_mean, np.asarray(sfc_feat, np.float32).ravel()])[None]

    def _states_batch(self, Z_mean: np.ndarray, sfc_feats: np.ndarray) -> np.ndarray:
        """
        Z_mean: (latent,)  sfc_feats: (N, FEAT_PER_SFC)
        Trả về (N, latent + FEAT_PER_SFC) — pure numpy, không loop.
        """
        z = np.broadcast_to(Z_mean[None], (len(sfc_feats), len(Z_mean)))
        return np.concatenate([z, sfc_feats], axis=1).astype(np.float32)

    @staticmethod
    def _nondominated_sort(q_matrix: np.ndarray) -> List[int]:
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
        q_mat   = self.policy_net(S_batch, training=False).numpy()  # (N, 2)

        q_pareto = np.column_stack([q_mat[:, 0], -q_mat[:, 1]])
        front    = self._nondominated_sort(q_pareto)
        return max(front, key=lambda i: q_mat[i, 0])

    def train(self, buffer: ReplayBuffer, batch_size: int = 16):
        if len(buffer) < batch_size:
            return
        batch = buffer.sample(batch_size)

        states, next_states = [], []
        actions, rewards_ar, rewards_cost, dones = [], [], [], []

        for (Z_mean, sfc_feats, sfc_idx, R_HL, Z_next_mean, sfc_feats_next, done) in batch:
            if len(sfc_feats) == 0:
                continue
            idx_clip = min(int(sfc_idx), len(sfc_feats) - 1)

            z  = np.asarray(Z_mean,      np.float32).ravel()
            zn = np.asarray(Z_next_mean, np.float32).ravel()
            f  = np.asarray(sfc_feats[idx_clip], np.float32).ravel()
            fn = np.asarray(sfc_feats_next[0] if len(sfc_feats_next) > 0
                            else sfc_feats[idx_clip], np.float32).ravel()

            states.append(np.concatenate([z, f]))
            next_states.append(np.concatenate([zn, fn]))
            actions.append(idx_clip)

            r_ar   = float(R_HL[0]) if hasattr(R_HL, '__len__') else float(R_HL)
            r_cost = float(R_HL[1]) if hasattr(R_HL, '__len__') and len(R_HL) > 1 else 0.0
            rewards_ar.append(r_ar)
            rewards_cost.append(r_cost)
            dones.append(float(done))

        if not states:
            return

        S      = tf.constant(np.array(states,       np.float32))
        Sn     = tf.constant(np.array(next_states,  np.float32))
        R_ar   = tf.constant(np.array(rewards_ar,   np.float32))
        R_cost = tf.constant(np.array(rewards_cost, np.float32))
        D      = tf.constant(np.array(dones,        np.float32))
        A      = np.array(actions, dtype=np.int32)

        Q_next   = self.target_net(Sn, training=False)
        tgt_ar   = R_ar   + self.gamma * Q_next[:, 0] * (1.0 - D)
        tgt_cost = R_cost + self.gamma * Q_next[:, 1] * (1.0 - D)

        with tf.GradientTape() as tape:
            Q_pred    = self.policy_net(S, training=True)
            loss_ar   = tf.reduce_mean(tf.square(tgt_ar   - Q_pred[:, 0]))
            loss_cost = tf.reduce_mean(tf.square(tgt_cost - Q_pred[:, 1]))
            loss      = loss_ar + loss_cost
        self.opt.apply_gradients(
            zip(tape.gradient(loss, self.policy_net.trainable_variables),
                self.policy_net.trainable_variables))
