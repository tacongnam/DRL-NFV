import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import layers, models, optimizers
import config

def build_q_network():
    """Xây dựng và compile mô hình Q-Network đa đầu vào (Multi-input)."""
    # Input 1: DC Info
    in_shape_1 = (2 * config.NUM_VNF_TYPES + 2,)
    input_1 = layers.Input(shape=in_shape_1, name="Input_DC")

    # Input 2: DC-SFC Info
    in_shape_2 = (config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),)
    input_2 = layers.Input(shape=in_shape_2, name="Input_DC_SFC")

    # Input 3: Global Info
    in_shape_3 = (config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),)
    input_3 = layers.Input(shape=in_shape_3, name="Input_Global")

    # Lớp Dense cho từng input
    d1 = layers.Dense(32, activation='relu')(input_1)
    d2 = layers.Dense(64, activation='relu')(input_2)
    d3 = layers.Dense(64, activation='relu')(input_3)

    # Kết hợp các đặc trưng
    concat = layers.Concatenate()([d1, d2, d3])

    # Cơ chế Attention
    attn_weights = layers.Dense(160, activation='sigmoid')(concat)
    attended = layers.Multiply()([concat, attn_weights])

    # Các lớp Fully Connected tiếp theo
    fc1 = layers.Dense(96, activation='relu')(attended)
    fc1 = layers.Dropout(0.2)(fc1)
    fc2 = layers.Dense(64, activation='relu')(fc1)

    # Output Layer
    output = layers.Dense(config.ACTION_SPACE_SIZE, activation='linear', name="Output")(fc2)

    model = models.Model(inputs=[input_1, input_2, input_3], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='mse')
    
    return model