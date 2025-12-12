# agent/model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers, models, optimizers
import config

def build_q_network():
    """
    Xây dựng Q-Network với kiến trúc multi-input và attention mechanism
    
    Architecture:
    - Input 1: DC Info [2|V|+2]
    - Input 2: DC-SFC Info [|S|(1+2|V|)]
    - Input 3: Global SFC Info [|S|(4+|V|)]
    - Concatenate -> Attention -> FC layers -> Output [2|V|+1]
    """
    
    # --- Input Layers ---
    # Input 1: DC State
    input_1_shape = (2 * config.NUM_VNF_TYPES + 2,)
    input_1 = layers.Input(shape=input_1_shape, name="Input_DC_State")
    
    # Input 2: DC-SFC State
    input_2_shape = (config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),)
    input_2 = layers.Input(shape=input_2_shape, name="Input_DC_SFC_State")
    
    # Input 3: Global SFC State
    input_3_shape = (config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),)
    input_3 = layers.Input(shape=input_3_shape, name="Input_Global_SFC_State")
    
    # --- Processing Layers ---
    # Dense layers cho từng input
    dense_1 = layers.Dense(32, activation='relu', name="Dense_DC")(input_1)
    dense_2 = layers.Dense(64, activation='relu', name="Dense_DC_SFC")(input_2)
    dense_3 = layers.Dense(64, activation='relu', name="Dense_Global")(input_3)
    
    # --- Concatenation ---
    concat = layers.Concatenate(name="Concat")([dense_1, dense_2, dense_3])
    concat_dim = concat.shape[-1]
    
    # --- Attention Mechanism ---
    # Tính attention weights
    attn_size = int(concat_dim)
    attention_weights = layers.Dense(attn_size, activation='sigmoid', name="Attention_Weights")(concat)
    
    # Apply attention
    attended = layers.Multiply(name="Attention_Applied")([concat, attention_weights])
    
    # --- Fully Connected Layers ---
    fc1 = layers.Dense(96, activation='relu', name="FC1")(attended)
    fc1 = layers.LayerNormalization(name="Norm1")(fc1)
    fc2 = layers.Dense(64, activation='relu', name="FC2")(fc1)
    
    # --- Output Layer ---
    output = layers.Dense(config.ACTION_SPACE_SIZE, 
                         activation='linear', 
                         name="Q_Values")(fc2)
    
    # --- Build Model ---
    model = models.Model(inputs=[input_1, input_2, input_3], 
                        outputs=output,
                        name="DQN_SFC_Provisioning")
    
    # Compile
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=config.LEARNING_RATE, weight_decay=1e-4),
        loss='mse'
    )
    
    return model