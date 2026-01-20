import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers, models, optimizers
import config

def build_q_network():
    """
    Enhanced Q-Network with chain pattern encoding
    
    Features per chain: (max_length=4) + NUM_VNF_TYPES + 3 stats = 4 + V + 3
    
    Architecture:
    - Input 1: DC State [2 + 2*V]
    - Input 2: DC Demand [V + 3*(4+V+3)] = [V + 3*(7+V)]
    - Input 3: Global State [4 + V + 5*(4+V+3)] = [4 + V + 5*(7+V)]
    - Output: Q-values [2*V + 1]
    """
    
    V = config.NUM_VNF_TYPES
    chain_feature_size = 4 + V + 3  # 7 + V
    
    # --- Input Layers ---
    input_1_shape = (2 + 2 * V,)
    input_1 = layers.Input(shape=input_1_shape, name="Input_DC_State")
    
    input_2_shape = (V + 3 * chain_feature_size,)
    input_2 = layers.Input(shape=input_2_shape, name="Input_DC_Demand")
    
    input_3_shape = (4 + V + 5 * chain_feature_size,)
    input_3 = layers.Input(shape=input_3_shape, name="Input_Global_State")
    
    # --- Processing Layers ---
    dense_1 = layers.Dense(32, activation='relu', name="Dense_DC")(input_1)
    dense_2 = layers.Dense(64, activation='relu', name="Dense_DC_Demand")(input_2)
    dense_3 = layers.Dense(64, activation='relu', name="Dense_Global")(input_3)
    
    # --- Concatenation ---
    concat = layers.Concatenate(name="Concat")([dense_1, dense_2, dense_3])
    concat_dim = concat.shape[-1]
    
    # --- Attention Mechanism ---
    attention_weights = layers.Dense(int(concat_dim), activation='sigmoid', name="Attention_Weights")(concat)
    attended = layers.Multiply(name="Attention_Applied")([concat, attention_weights])
    
    # --- Fully Connected Layers ---
    fc1 = layers.Dense(96, activation='relu', name="FC1")(attended)
    fc1 = layers.LayerNormalization(name="Norm1")(fc1)
    fc2 = layers.Dense(64, activation='relu', name="FC2")(fc1)
    
    # --- Output Layer ---
    output = layers.Dense(config.get_action_space_size(), 
                         activation='linear', 
                         name="Q_Values")(fc2)
    
    # --- Build Model ---
    model = models.Model(inputs=[input_1, input_2, input_3], 
                        outputs=output,
                        name="DQN_SFC_Provisioning")
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=config.LEARNING_RATE, weight_decay=1e-4),
        loss='mse'
    )
    
    return model