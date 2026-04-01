import tensorflow as tf
import numpy as np
import random
from typing import List


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_features):
        super().__init__()
        self.dense = tf.keras.layers.Dense(out_features)

    def call(self, X, A):
        return self.dense(tf.matmul(A, X))


class VGAE_NN(tf.keras.Model):
    def __init__(self, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.base_gcn   = GCNLayer(hidden_dim)
        self.gcn_mu     = GCNLayer(latent_dim)
        self.gcn_logvar = GCNLayer(latent_dim)

    def call(self, X, A):
        h = tf.nn.relu(self.base_gcn(X, A))
        return self.gcn_mu(h, A), self.gcn_logvar(h, A)

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        return mu + tf.random.normal(tf.shape(std)) * std


class VGAENetwork:
    def __init__(self):
        self.model     = VGAE_NN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def normalize_adj(self, adj):
        adj = adj + np.eye(adj.shape[0])
        d   = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D = np.diag(d_inv_sqrt)
        return adj.dot(D).transpose().dot(D)

    def encode(self, X, A):
        X_t   = tf.convert_to_tensor(X, dtype=tf.float32)
        A_t   = tf.convert_to_tensor(self.normalize_adj(A), dtype=tf.float32)
        mu, logvar = self.model(X_t, A_t)
        return self.model.reparameterize(mu, logvar).numpy()

    def train(self, buffer, epochs=5):
        for _ in range(epochs):
            batch = buffer.sample(16)
            for X, A in batch:
                X_t = tf.convert_to_tensor(X, dtype=tf.float32)
                A_t = tf.convert_to_tensor(self.normalize_adj(A), dtype=tf.float32)
                A_target = tf.convert_to_tensor(A, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    mu, logvar = self.model(X_t, A_t)
                    Z     = self.model.reparameterize(mu, logvar)
                    A_hat = tf.sigmoid(tf.matmul(Z, Z, transpose_b=True))
                    bce   = tf.reduce_mean(tf.keras.losses.binary_crossentropy(A_target, A_hat))
                    kl    = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)) \
                            / tf.cast(tf.shape(X_t)[0], tf.float32)
                    loss  = bce + kl

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class PMDRL_NN(tf.keras.Sequential):
    def __init__(self, input_dim: int = 12):
        super().__init__([
            tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(2)
        ])


class HighLevelAgent:
    def __init__(self, gamma=0.95, use_ll_score: bool = False, input_dim: int = 12):
        self.policy_net = PMDRL_NN(input_dim=input_dim)
        self.target_net = PMDRL_NN(input_dim=input_dim)
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.gamma      = gamma
        self.use_ll_score = use_ll_score

    def compute_ll_scores(self, waitlist, Z_t, ll_agent: 'LowLevelAgent') -> List[float]:
        scores = []
        n_dcs = Z_t.shape[0]
        
        for sfc in waitlist:
            sfc_score = 0.0
            for vnf in sfc.request.vnfs:
                vnf_feat = np.array([[vnf.resource['mem'], vnf.resource['cpu'], vnf.resource['ram']]], dtype=np.float32)
                inputs = np.concatenate([Z_t, np.tile(vnf_feat, [n_dcs, 1])], axis=1)
                q_values = ll_agent.policy_net(inputs).numpy()
                sfc_score += float(np.max(q_values))
            scores.append(sfc_score)
        
        return scores

    def extract_sfc_features(self, waitlist, Z_t=None, ll_agent=None):
        feats = []
        ll_scores = []
        
        if self.use_ll_score and Z_t is not None and ll_agent is not None:
            ll_scores = self.compute_ll_scores(waitlist, Z_t, ll_agent)
        
        for i, sfc in enumerate(waitlist):
            total_cpu = sum(v.resource['cpu'] for v in sfc.request.vnfs)
            feat = [len(sfc.request.vnfs), total_cpu, sfc.request.bw]
            if self.use_ll_score:
                feat.append(ll_scores[i] if i < len(ll_scores) else 0.0)
            feats.append(feat)
        
        return np.array(feats, dtype=np.float32)

    def get_pareto_front(self, q_vectors):
        n = len(q_vectors)
        pareto = []
        for i in range(n):
            dominated = any(
                j != i
                and q_vectors[j][0] >= q_vectors[i][0]
                and q_vectors[j][1] >= q_vectors[i][1]
                and (q_vectors[j][0] > q_vectors[i][0] or q_vectors[j][1] > q_vectors[i][1])
                for j in range(n)
            )
            if not dominated:
                pareto.append(i)
        return pareto if pareto else [0]

    def _build_inputs(self, Z_mean, sfc_feats):
        return np.concatenate([np.tile(Z_mean, [len(sfc_feats), 1]), sfc_feats], axis=1)

    def act(self, Z, waitlist, epsilon, ll_agent=None):
        sfc_feats = self.extract_sfc_features(
            waitlist,
            Z_t=Z if self.use_ll_score else None,
            ll_agent=ll_agent if self.use_ll_score else None
        )
        Z_mean    = np.mean(Z, axis=0, keepdims=True)
        inputs    = self._build_inputs(Z_mean, sfc_feats)
        q_vectors = self.policy_net(inputs).numpy()
        pareto    = self.get_pareto_front(q_vectors)

        if random.random() < epsilon:
            return random.randint(0, len(waitlist) - 1)
        return random.choice(pareto)

    def train(self, buffer, batch_size=32):
        batch     = buffer.sample(batch_size)
        loss_val  = tf.constant(0.0)

        with tf.GradientTape() as tape:
            for (Z_mean, sfc_feats, a, R_HL, Z_next_mean, sfc_feats_next, done) in batch:
                if len(sfc_feats) == 0:
                    continue

                inputs  = tf.constant(self._build_inputs(Z_mean, sfc_feats))
                q_vals  = self.policy_net(inputs)
                q_action = q_vals[a]

                if done or len(sfc_feats_next) == 0:
                    y_target = tf.convert_to_tensor(R_HL, dtype=tf.float32)
                else:
                    inp_next    = tf.constant(self._build_inputs(Z_next_mean, sfc_feats_next))
                    q_next      = self.target_net(inp_next).numpy()
                    pareto_idx  = self.get_pareto_front(q_next)
                    expected_q  = np.mean(q_next[pareto_idx], axis=0)
                    y_target    = tf.convert_to_tensor(R_HL, dtype=tf.float32) \
                                  + self.gamma * tf.convert_to_tensor(expected_q, dtype=tf.float32)

                loss_val = loss_val + tf.reduce_mean(tf.square(y_target - q_action))

            loss_val = loss_val / float(len(batch))

        grads = tape.gradient(loss_val, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())


class DQN_NN(tf.keras.Sequential):
    def __init__(self, input_dim: int = 11):
        super().__init__([
            tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(1)
        ])


class LowLevelAgent:
    def __init__(self, gamma=0.95, input_dim: int = 11):
        self.policy_net = DQN_NN(input_dim=input_dim)
        self.target_net = DQN_NN(input_dim=input_dim)
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.gamma      = gamma

    def _masked_q(self, q_values: np.ndarray, valid_mask) -> np.ndarray:
        q = q_values.copy().astype(float)
        mask = np.ones(len(q), dtype=bool)
        mask[valid_mask] = False
        q[mask] = -np.inf
        return q

    def act(self, Z, vnf_req, valid_mask, epsilon):
        if random.random() < epsilon:
            return random.choice(valid_mask)

        inputs   = np.concatenate([Z, np.tile([vnf_req], [Z.shape[0], 1])], axis=1)
        q_values = self.policy_net(inputs).numpy()
        return int(np.argmax(self._masked_q(q_values, valid_mask)))

    def train(self, buffer, batch_size=32):
        batch    = buffer.sample(batch_size)
        loss_val = tf.constant(0.0)

        with tf.GradientTape() as tape:
            for (Z_t, vnf_feat, a, R_LL, Z_next, valid_mask_next, done) in batch:
                num_nodes = Z_t.shape[0]
                inputs    = tf.concat([Z_t, tf.tile(vnf_feat, [num_nodes, 1])], axis=1)
                q_vals    = self.policy_net(inputs)
                q_action  = q_vals[a]

                if done or len(valid_mask_next) == 0:
                    y_target = tf.constant(R_LL, dtype=tf.float32)
                else:
                    inp_next    = tf.concat([Z_next, tf.tile(vnf_feat, [num_nodes, 1])], axis=1)
                    q_next      = self.target_net(inp_next).numpy()
                    max_q       = float(np.max(self._masked_q(q_next, valid_mask_next)))
                    y_target    = tf.constant(R_LL + self.gamma * max_q, dtype=tf.float32)

                loss_val = loss_val + tf.square(y_target - q_action)

            loss_val = loss_val / float(len(batch))

        grads = tape.gradient(loss_val, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())