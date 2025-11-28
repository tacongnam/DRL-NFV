import tensorflow as tf
from keras import layers, models, Model
import config

def create_dqn_model(num_actions):
    """
    [Paper Fig. 1 - Right Side]: Multi-input DQN with Attention
    Inputs:
    1. DC State (l1xd): Tài nguyên DC (CPU, RAM...)
    2. DC's SFC State (l2xd): Các VNF đang chạy trên DC này
    3. Overall SFC State (l3xd): Yêu cầu SFC toàn cục (BW, Delay...)
    """
    
    # --- Input 1: DC Resources State ---
    input_dc = layers.Input(shape=(3,), name="dc_state_input") # [CPU, RAM, Stor]
    x1 = layers.Dense(32, activation='relu')(input_dc)
    
    # --- Input 2: DC's SFC State ---
    # (Ví dụ: One-hot vector các loại VNF đang có trên DC)
    input_dc_sfc = layers.Input(shape=(config.NUM_VNFS,), name="dc_sfc_input")
    x2 = layers.Dense(32, activation='relu')(input_dc_sfc)
    
    # --- Input 3: Overall SFC State ---
    # (BW, Delay, Rem_VNFs, Current_VNF_Type)
    input_global = layers.Input(shape=(4,), name="global_sfc_input")
    x3 = layers.Dense(32, activation='relu')(input_global)
    
    # --- Feature Fusion ---
    # Ghép các đặc trưng lại
    concat = layers.Concatenate()([x1, x2, x3])
    
    # --- Attention Mechanism ---
    # Tạo attention weights để model biết nên tập trung vào input nào
    attention_probs = layers.Dense(concat.shape[-1], activation='softmax', name='attention_probs')(concat)
    attention_mul = layers.Multiply(name='attention_mul')([concat, attention_probs])
    
    # --- FCDNN (Fully Connected) ---
    x = layers.Dense(128, activation='relu')(attention_mul)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output: Q-Values for Actions
    output = layers.Dense(num_actions, activation='linear')(x)
    
    model = Model(inputs=[input_dc, input_dc_sfc, input_global], outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.DQN_LR),
                  loss='mse')
    return model