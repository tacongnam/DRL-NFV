from __future__ import annotations
import os, random, collections
from typing import Optional
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from keras import layers


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
        self.act = layers.Activation(activation) if activation else None

    def call(self, H, A_hat):
        out = self.dense(tf.matmul(A_hat, H))
        return self.act(out) if self.act else out

class VGAENetwork:
    NODE_FEAT_DIM = 8

    def __init__(self, hidden_dim: int = 16, latent_dim: int = 8,
                 lr: float = 1e-3, beta: float = 1e-3, lambda_aux: float = 0.5):
        self.latent_dim = latent_dim
        self.beta = beta
        self.lambda_aux = lambda_aux
        self._built = False
        self._adj_cache: dict = {}
        self.gcn1 = GCNLayer(hidden_dim, activation="relu", name="gcn1")
        self.gcn_mu = GCNLayer(latent_dim, activation=None, name="gcn_mu")
        self.gcn_lv = GCNLayer(latent_dim, activation=None, name="gcn_logvar")
        self.aux_head = layers.Dense(self.NODE_FEAT_DIM, name="aux_head")
        self.optimizer = keras.optimizers.Adam(lr)

    def _norm_adj(self, A: np.ndarray) -> tf.Tensor:
        key = A.tobytes()
        if key in self._adj_cache:
            return self._adj_cache[key]
        N = A.shape[0]
        Ai = A + np.eye(N, dtype=np.float32)
        deg = Ai.sum(axis=1, keepdims=True).clip(min=1)
        D = np.diag(1.0 / np.sqrt(deg.flatten()))
        result = tf.constant(D @ Ai @ D, dtype=tf.float32)
        if len(self._adj_cache) < 256:
            self._adj_cache[key] = result
        return result

    @tf.function(reduce_retracing=True)
    def _gcn_forward(self, X_t, A_hat):
        h = self.gcn1(X_t, A_hat)
        return self.gcn_mu(h, A_hat), self.gcn_lv(h, A_hat)

    def encode(self, X: np.ndarray, A: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        X_t = tf.constant(X, dtype=tf.float32)
        A_hat = self._norm_adj(A.astype(np.float32))
        mu, log_var = self._gcn_forward(X_t, A_hat)
        self._built = True
        if deterministic:
            return mu.numpy()
        lv = tf.clip_by_value(log_var, -10.0, 10.0)
        return (mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))).numpy()

    def _train_step(self, X_t, A_hat, A_t):
        with tf.GradientTape() as tape:
            h = self.gcn1(X_t, A_hat)
            mu = self.gcn_mu(h, A_hat)
            lv = tf.clip_by_value(self.gcn_lv(h, A_hat), -10.0, 10.0)
            z = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
            A_pred = tf.sigmoid(tf.matmul(z, tf.transpose(z)))
            eps_ = 1e-7
            bce = -tf.reduce_mean(
                A_t * tf.math.log(A_pred + eps_) +
                (1 - A_t) * tf.math.log(1 - A_pred + eps_))
            kl = -0.5 * tf.reduce_mean(1 + lv - mu ** 2 - tf.exp(lv))
            vgae_loss = bce + self.beta * kl
            aux = tf.reduce_mean(tf.square(self.aux_head(z) - X_t))
            lam = self.lambda_aux * vgae_loss / (vgae_loss + aux + 1e-7)
            loss = vgae_loss + lam * aux
        vs = (self.gcn1.trainable_variables + self.gcn_mu.trainable_variables +
              self.gcn_lv.trainable_variables + self.aux_head.trainable_variables)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, vs), vs))
        return loss.numpy()

    def train(self, buffer: ReplayBuffer, epochs: int = 1, batch: int = 16):
        if len(buffer) < 4:
            return None
        total, count = 0.0, 0
        for _ in range(epochs):
            for X, A in buffer.sample(batch):
                if X.shape[0] < 2:
                    continue
                loss = self._train_step(
                    tf.constant(X, dtype=tf.float32),
                    self._norm_adj(A.astype(np.float32)),
                    tf.constant(A, dtype=tf.float32))
                total += loss
                count += 1
        return total / count if count > 0 else None

    def _ensure_built(self):
        if not self._built:
            self.encode(np.zeros((2, self.NODE_FEAT_DIM), np.float32),
                        np.eye(2, dtype=np.float32))
            self.aux_head(tf.zeros((2, self.latent_dim)))

    def save_weights(self, path: str):
        self._ensure_built()
        np.save(path, {
            "gcn1": [v.numpy() for v in self.gcn1.trainable_variables],
            "gcn_mu": [v.numpy() for v in self.gcn_mu.trainable_variables],
            "gcn_lv": [v.numpy() for v in self.gcn_lv.trainable_variables],
            "aux_head": [v.numpy() for v in self.aux_head.trainable_variables],
        }, allow_pickle=True)

    def load_weights(self, path: str):
        try:
            w = np.load(path, allow_pickle=True).item()
            self._ensure_built()
            for layer, key in [(self.gcn1, "gcn1"), (self.gcn_mu, "gcn_mu"),
                               (self.gcn_lv, "gcn_lv"), (self.aux_head, "aux_head")]:
                if key in w:
                    for var, val in zip(layer.trainable_variables, w[key]):
                        var.assign(val)
        except Exception as e:
            print(f"[VGAE] Could not load weights: {e}")


def _mlp(input_dim: int, hidden: int, out_dim: int, name: str) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name=name + "_in")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    return keras.Model(inp, layers.Dense(out_dim)(x), name=name)