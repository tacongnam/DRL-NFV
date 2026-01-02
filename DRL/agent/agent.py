import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

<<<<<<< Updated upstream:DRL/agent/agent.py
import tensorflow as tf
import numpy as np
import config
from agent.model import build_q_network
=======
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers
>>>>>>> Stashed changes:agents/dqn_agent.py
from collections import deque
import config

class DQNAgent:
    """DQN Agent for SFC Placement"""
    
<<<<<<< Updated upstream:DRL/agent/agent.py
    def __init__(self):
        self.model = build_q_network()
        self.target_model = build_q_network()
        self.update_target_model()

        self.batch_size = config.BATCH_SIZE
        self.optimizer = self.model.optimizer      # AdamW
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.global_replay_memory = deque(maxlen=config.MEMORY_SIZE)

        # Pre-allocated buffer để giảm allocate lại
        self.batch_size = config.BATCH_SIZE

    def reset_global_replay_memory(self):
        self.global_replay_memory = deque(maxlen=config.MEMORY_SIZE)
=======
    def __init__(self, state_shapes, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # Build networks
        self.q_network = self._build_network(state_shapes, action_size)
        self.target_network = self._build_network(state_shapes, action_size)
        self.update_target_network()
        
        self.optimizer = optimizers.AdamW(
            learning_rate=config.LEARNING_RATE, 
            weight_decay=1e-4
        )
>>>>>>> Stashed changes:agents/dqn_agent.py
    
    def _build_network(self, state_shapes, action_size):
        """Build Q-Network with 3 inputs (DC, Demand, Global)"""        
        # Inputs
        input_dc = layers.Input(shape=state_shapes[0], name="dc_state")
        input_dc_demand = layers.Input(shape=state_shapes[1], name="dc_demand")
        input_global = layers.Input(shape=state_shapes[2], name="global_state")
        
        # Feature extraction
        x1 = layers.Dense(32, activation='relu')(input_dc)
        x2 = layers.Dense(64, activation='relu')(input_dc_demand)
        x3 = layers.Dense(64, activation='relu')(input_global)
        
        # Fusion & Attention
        concat = layers.Concatenate()([x1, x2, x3])
        attn = layers.Dense(concat.shape[-1], activation='sigmoid')(concat)
        x = layers.Multiply()([concat, attn])
        
        # Decision layers
        x = layers.Dense(96, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Q-values
        q_values = layers.Dense(action_size, activation='linear')(x)
        
        return models.Model(
            inputs=[input_dc, input_dc_demand, input_global],
            outputs=q_values
        )
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state, epsilon, valid_mask=None):
        """
        Epsilon-greedy action selection
        
        Args:
            state: Tuple (s1, s2, s3) - each is 1D numpy array
            epsilon: Exploration rate
            valid_mask: Boolean array of valid actions
        
        Returns:
            int: Selected action
        """
        # Exploration
        if np.random.rand() < epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            return np.random.randint(self.action_size)
        
        # Exploitation
        inputs = [np.expand_dims(s, 0) for s in state]
        
        q_values = self.q_network(inputs, training=False)[0].numpy()
        
        if valid_mask is not None:
            q_values = np.where(valid_mask, q_values, -np.inf)
        
        return np.argmax(q_values)
    
<<<<<<< Updated upstream:DRL/agent/agent.py
    @tf.function
    def train_step(self, state_batch, action_batch, reward_batch,
                   next_state_batch, done_batch):
        """
        state_batch       : [3 tensor]
        next_state_batch  : [3 tensor]
        action_batch      : int32 tensor
        reward_batch      : float32
        """

        # ------------------------------------------
        # Compute target Q-values
        # ------------------------------------------
        next_q = self.target_model(next_state_batch, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)

        targets = reward_batch + (1.0 - tf.cast(done_batch, tf.float32)) * \
                  config.GAMMA * max_next_q

        # ------------------------------------------
        # Train main network
        # ------------------------------------------
        with tf.GradientTape() as tape:
            q_values = self.model(state_batch, training=True)

            # Gather Q(s,a)
            batch_indices = tf.range(self.batch_size, dtype=tf.int32)
            action_indices = tf.stack([batch_indices, action_batch], axis=1)
            predicted = tf.gather_nd(q_values, action_indices)

            loss = self.loss_fn(targets, predicted)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    # -------------------------------------------------
    #  Wrapper train — chuẩn bị batch rồi gọi graph
    # -------------------------------------------------
    def train(self):
        if len(self.global_replay_memory) < self.batch_size:
=======
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=None):
        """Train on batch from replay buffer"""
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        if len(self.memory) < batch_size:
>>>>>>> Stashed changes:agents/dqn_agent.py
            return 0.0
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = [[], [], []]
        next_states = [[], [], []]
        actions, rewards, dones = [], [], []
        
        for s, a, r, ns, d in batch:
            for i in range(3):
                states[i].append(s[i])
                next_states[i].append(ns[i])
            actions.append(a)
            rewards.append(r)
            dones.append(d)
        
        # Convert to tensors
        states_t = [tf.convert_to_tensor(np.array(x), dtype=tf.float32) for x in states]
        next_states_t = [tf.convert_to_tensor(np.array(x), dtype=tf.float32) for x in next_states]
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train step
        loss = self._train_step(states_t, actions_t, rewards_t, next_states_t, dones_t)
        return float(loss.numpy())
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """Single training step (graph mode)"""
        # Compute target Q-values
        next_q = self.target_network(next_states, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + (1.0 - dones) * config.GAMMA * max_next_q
        
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            
            batch_indices = tf.range(tf.shape(actions)[0])
            action_indices = tf.stack([batch_indices, actions], axis=1)
            predicted_q = tf.gather_nd(q_values, action_indices)
            
            loss = tf.reduce_mean(tf.square(targets - predicted_q))
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss
    
    def save(self, path):
        self.q_network.save_weights(f"{path}_q.weights.h5")
        self.target_network.save_weights(f"{path}_target.weights.h5")
    
    def load(self, path):
        self.q_network.load_weights(f"{path}_q.weights.h5")
        self.target_network.load_weights(f"{path}_target.weights.h5")