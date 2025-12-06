import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras import layers, models, optimizers
import config
import numpy as np
import random 

class AttentionLayer(layers.Layer):
    """Custom Attention Layer đơn giản để highlight features"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True) 
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, features)
        e = tf.keras.activations.tanh(tf.matmul(x, self.W))
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return output

class SFC_DQN:
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Input 1: DC Info
        in_shape_1 = (2 * config.NUM_VNF_TYPES + 2,)
        input_1 = layers.Input(shape=in_shape_1, name="Input_DC")

        # Input 2: DC-SFC Info
        in_shape_2 = (config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),)
        input_2 = layers.Input(shape=in_shape_2, name="Input_DC_SFC")

        # Input 3: Global Info
        in_shape_3 = (config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),)
        input_3 = layers.Input(shape=in_shape_3, name="Input_Global")

        # Feature extraction
        d1 = layers.Dense(64, activation='relu')(input_1)
        d2 = layers.Dense(128, activation='relu')(input_2)
        d3 = layers.Dense(128, activation='relu')(input_3)

        # Concatenate
        concat = layers.Concatenate()([d1, d2, d3])

        # Attention Layer
        attn = layers.Dense(320, activation='sigmoid')(concat) 
        multiplied = layers.Multiply()([concat, attn])

        # Fully Connected Layers
        fc1 = layers.Dense(256, activation='relu')(multiplied)
        fc2 = layers.Dense(128, activation='relu')(fc1)

        # Output Layer
        output = layers.Dense(config.ACTION_SPACE_SIZE, activation='linear', name="Output")(fc2)

        model = models.Model(inputs=[input_1, input_2, input_3], outputs=output)
        model.compile(optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, replay_memory):
        if len(replay_memory) < config.BATCH_SIZE:
            return 0

        minibatch = random.sample(replay_memory, config.BATCH_SIZE)
        
        # Prepare batches
        states_1 = np.array([i[0][0] for i in minibatch])
        states_2 = np.array([i[0][1] for i in minibatch])
        states_3 = np.array([i[0][2] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states_1 = np.array([i[3][0] for i in minibatch])
        next_states_2 = np.array([i[3][1] for i in minibatch])
        next_states_3 = np.array([i[3][2] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Q-Value targets
        target_qs = self.target_model.predict([next_states_1, next_states_2, next_states_3], verbose=0)
        targets = self.model.predict([states_1, states_2, states_3], verbose=0)

        for i in range(config.BATCH_SIZE):
            target_val = rewards[i]
            if not dones[i]:
                target_val += config.GAMMA * np.max(target_qs[i])
            targets[i][actions[i]] = target_val

        history = self.model.fit(
            [states_1, states_2, states_3], targets, 
            batch_size=config.BATCH_SIZE, verbose=0, epochs=1
        )
        return history.history['loss'][0]

    def get_action(self, state, epsilon, valid_actions_mask=None):
        """
        CRITICAL IMPROVEMENT: Action Masking
        - valid_actions_mask: Boolean array indicating which actions are valid
        - This prevents the model from even considering invalid actions
        """
        if np.random.rand() <= epsilon:
            # Random exploration - but only among VALID actions
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(config.ACTION_SPACE_SIZE)
        
        # Exploitation - get Q-values
        state_1 = tf.convert_to_tensor(state[0].reshape(1, -1))
        state_2 = tf.convert_to_tensor(state[1].reshape(1, -1))
        state_3 = tf.convert_to_tensor(state[2].reshape(1, -1))
        
        q_values = self.model([state_1, state_2, state_3], training=False)
        q_values = q_values.numpy()[0]
        
        # Apply action masking
        if valid_actions_mask is not None:
            # Set invalid actions to very negative value
            masked_q_values = np.where(valid_actions_mask, q_values, -1e9)
            return int(np.argmax(masked_q_values))
        
        return int(np.argmax(q_values))