import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from keras import layers, models, optimizers, activations
import config

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
        e = activations.tanh(tf.matmul(x, self.W))
        a = activations.softmax(e, axis=1)
        output = x * a
        return output

def build_q_network():
    """
    Xây dựng và compile mô hình Q-Network đa đầu vào (Multi-input).
    """
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

    # Concatenate features from all inputs
    concat = layers.Concatenate()([d1, d2, d3])

    # Attention Layer to focus on important features
    attn = layers.Dense(320, activation='sigmoid')(concat) 
    multiplied = layers.Multiply()([concat, attn])

    # Fully Connected Layers
    fc1 = layers.Dense(256, activation='relu')(multiplied)
    fc2 = layers.Dense(128, activation='relu')(fc1)

    # Output Layer (Q-values for each action)
    output = layers.Dense(config.ACTION_SPACE_SIZE, activation='linear', name="Output")(fc2)

    model = models.Model(inputs=[input_1, input_2, input_3], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='mse')
    
    return model