import tensorflow as tf
from keras import layers, Model
from config import *

class DQNAgent:
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR_DQN)
        self.epsilon = EPSILON_START

    def _build_model(self):
        # 1. DC State Input
        in_dc = layers.Input(shape=(STATE_DIM_DC,))
        x1 = layers.Dense(32, activation='relu')(in_dc)
        
        # 2. Local SFC Input
        in_loc = layers.Input(shape=(4,)) # [id, bw, delay, hops]
        x2 = layers.Dense(16, activation='relu')(in_loc)
        
        # 3. Global SFC Input
        in_glob = layers.Input(shape=(len(SFC_PROFILES),))
        x3 = layers.Dense(16, activation='relu')(in_glob)
        
        # Concatenate
        concat = layers.Concatenate()([x1, x2, x3])
        
        # Attention Mechanism (Simple)
        att = layers.Dense(64, activation='softmax')(concat)
        mul = layers.Multiply()([concat, att])
        
        h = layers.Dense(64, activation='relu')(mul)
        h = layers.Dense(32, activation='relu')(h)
        out = layers.Dense(ACTION_SPACE, activation='linear')(h)
        
        return Model(inputs=[in_dc, in_loc, in_glob], outputs=out)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, dc_s, loc_s, glob_s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(ACTION_SPACE)
        
        q_vals = self.model([dc_s[np.newaxis], loc_s[np.newaxis], glob_s[np.newaxis]])
        return np.argmax(q_vals[0])

    def train(self, batch):
        dc_s, loc_s, glob_s, act, rew, n_dc_s, n_loc_s, n_glob_s, done = batch
        
        with tf.GradientTape() as tape:
            # Current Q
            q_values = self.model([dc_s, loc_s, glob_s])
            indices = tf.range(BATCH_SIZE, dtype=tf.int32)
            act_indices = tf.stack([indices, tf.cast(act, tf.int32)], axis=1)
            chosen_q = tf.gather_nd(q_values, act_indices)
            
            # Target Q
            next_q = self.target_model([n_dc_s, n_loc_s, n_glob_s])
            max_next_q = tf.reduce_max(next_q, axis=1)
            targets = rew + (1 - done) * GAMMA * max_next_q
            
            loss = tf.reduce_mean(tf.square(targets - chosen_q))
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        return loss