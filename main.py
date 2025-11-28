import numpy as np
import tensorflow as tf
from tqdm import tqdm  # [NEW] Import tqdm
import config
from env.core_network import CoreNetworkEnv
from env.traffic_generator import TrafficGenerator
from models.vae_model import VAEWithValueHead
from models.dqn_model import create_dqn_model
from utils.buffers import ReplayBuffer, VAEDataset

# --- INITIALIZATION ---
env = CoreNetworkEnv()
traffic_gen = TrafficGenerator()

# Models
vae = VAEWithValueHead(input_dim=3) 
vae_opt = tf.keras.optimizers.Adam(learning_rate=config.VAE_LR)

# Lưu ý: Action space size tính toán dựa trên config
# 2 * NUM_VNFS + 1 (Wait)
dqn = create_dqn_model(num_actions=env.action_space.n)

# Buffers
dqn_buff = ReplayBuffer(config.MEMORY_SIZE)
vae_buff = VAEDataset()

EPSILON = 1.0
DECAY = 0.995
NUM_EPISODES = 5
REQUESTS_PER_EPISODE = 10

# --- MAIN LOOP WITH TQDM ---
# Wrap range() bằng tqdm để tạo thanh tiến trình
with tqdm(range(NUM_EPISODES), desc="Training Phase", unit="ep") as pbar:
    for episode in pbar:
        # Reset Environment
        _ = env.reset() 
        total_ep_reward = 0
        
        # Tracking losses cho hiển thị
        ep_dqn_losses = []
        ep_vae_losses = []
        
        for _ in range(REQUESTS_PER_EPISODE):
            sfc_type, sfc_data = traffic_gen.generate_request()
            env.set_current_request(sfc_data)
            request_active = True
            
            while request_active:
                used_dcs = env.placed_vnfs_loc
                
                # --- 1. DC SELECTION (GenAI or Random) ---
                best_dc_idx = -1
                
                # Epsilon Exploration cho chọn DC
                if np.random.rand() < EPSILON:
                    candidates = [i for i in range(config.NUM_DCS) if i not in used_dcs]
                    if not candidates: 
                        request_active = False
                        break
                    best_dc_idx = np.random.choice(candidates)
                else:
                    # Use GenAI (Max Value)
                    max_val = -float('inf')
                    valid_dc_found = False
                    for dc_idx in range(config.NUM_DCS):
                        if dc_idx in used_dcs: continue
                        
                        valid_dc_found = True
                        dc_state = env.get_dc_state(dc_idx)
                        val = vae.get_dc_score(dc_state)
                        
                        if val > max_val:
                            max_val = val
                            best_dc_idx = dc_idx
                    
                    if not valid_dc_found:
                        request_active = False
                        break

                # Prepare DQN Inputs
                state_dict = env._get_state(selected_dc_idx=best_dc_idx)
                dc_state_curr = env.get_dc_state(best_dc_idx)

                # --- 2. DRL ACTION ---
                if np.random.rand() < EPSILON:
                    action = env.action_space.sample()
                else:
                    inp = [
                        state_dict["dc_state_input"][np.newaxis],
                        state_dict["dc_sfc_input"][np.newaxis],
                        state_dict["global_sfc_input"][np.newaxis]
                    ]
                    q_values = dqn.predict(inp, verbose=0)
                    action = np.argmax(q_values[0])
                
                # --- 3. EXECUTE & GET NEXT STATE ---
                next_state_dict, reward, done, info = env.step(action, best_dc_idx)
                dc_state_next = env.get_dc_state(best_dc_idx)
                
                # --- 4. STORE DATA ---
                dqn_buff.push(state_dict, action, reward, next_state_dict, done)
                
                if reward > -1.0: 
                    vae_buff.add(dc_state_curr, dc_state_next, reward)
                
                total_ep_reward += reward
                if done: request_active = False
                
                # --- 5. TRAINING ---
                # Train DQN
                if dqn_buff.size() > config.BATCH_SIZE:
                    s_batch, a, r, ns_batch, d = dqn_buff.sample(config.BATCH_SIZE)
                    
                    s_in1 = np.array([x['dc_state_input'] for x in s_batch])
                    s_in2 = np.array([x['dc_sfc_input'] for x in s_batch])
                    s_in3 = np.array([x['global_sfc_input'] for x in s_batch])
                    
                    ns_in1 = np.array([x['dc_state_input'] for x in ns_batch])
                    ns_in2 = np.array([x['dc_sfc_input'] for x in ns_batch])
                    ns_in3 = np.array([x['global_sfc_input'] for x in ns_batch])
                    
                    target_q = dqn.predict([ns_in1, ns_in2, ns_in3], verbose=0)
                    max_target = np.max(target_q, axis=1)
                    targets = r + config.GAMMA * max_target * (1 - d)
                    
                    curr_q = dqn.predict([s_in1, s_in2, s_in3], verbose=0)
                    indices = np.arange(config.BATCH_SIZE)
                    curr_q[indices, a] = targets
                    
                    # Train và lấy loss
                    history = dqn.fit([s_in1, s_in2, s_in3], curr_q, epochs=1, verbose=0)
                    ep_dqn_losses.append(history.history['loss'][0])
                
                # Train VAE
                if len(vae_buff.current_states) > config.BATCH_SIZE:
                    curr_b, next_b, val_b = vae_buff.get_batch(config.BATCH_SIZE)
                    curr_b = tf.convert_to_tensor(curr_b, dtype=tf.float32)
                    next_b = tf.convert_to_tensor(next_b, dtype=tf.float32)
                    val_b = tf.convert_to_tensor(val_b, dtype=tf.float32)
                    
                    with tf.GradientTape() as tape:
                        pred_next, pred_val, mean, log_var = vae(curr_b)
                        
                        recon_loss = tf.reduce_mean(tf.keras.losses.mse(next_b, pred_next))
                        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
                        value_loss = tf.reduce_mean(tf.keras.losses.mse(val_b, pred_val))
                        
                        total_loss = recon_loss + 0.01 * kl_loss + value_loss
                    
                    grads = tape.gradient(total_loss, vae.trainable_weights)
                    vae_opt.apply_gradients(zip(grads, vae.trainable_weights))
                    ep_vae_losses.append(total_loss.numpy())

        # Decay Epsilon
        if EPSILON > 0.05: EPSILON *= DECAY
        
        # --- UPDATE TQDM BAR ---
        # Tính toán giá trị trung bình để hiển thị cho gọn
        avg_dqn_loss = np.mean(ep_dqn_losses) if ep_dqn_losses else 0.0
        avg_vae_loss = np.mean(ep_vae_losses) if ep_vae_losses else 0.0
        
        pbar.set_postfix({
            'Rwd': f'{total_ep_reward:.1f}',      # Total Reward
            'Eps': f'{EPSILON:.2f}',              # Epsilon
            'D_Loss': f'{avg_dqn_loss:.3f}',      # DQN Loss
            'V_Loss': f'{avg_vae_loss:.3f}'       # VAE Loss
        })