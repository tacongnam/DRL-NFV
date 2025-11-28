import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import config
from env.core_network import CoreNetworkEnv
from env.traffic_generator import TrafficGenerator
from models.vae_model import VAEWithValueHead
from models.dqn_model import create_dqn_model
from utils.buffers import ReplayBuffer, VAEDataset

# --- SETUP GLOBAL VARIABLES ---
env = CoreNetworkEnv()
traffic_gen = TrafficGenerator()

# Models
vae = VAEWithValueHead(input_dim=3) 
vae_opt = tf.keras.optimizers.Adam(learning_rate=config.VAE_LR)
dqn = create_dqn_model(env.action_space.n)

# Buffers (Chỉ dùng cho training)
dqn_buff = ReplayBuffer(config.MEMORY_SIZE)
vae_buff = VAEDataset()

def run_episode(episode_idx, mode='train', epsilon=0.0):
    """
    Hàm chạy 1 episode. Dùng chung cho cả Train và Test để đảm bảo logic thống nhất.
    mode: 'train' (có học, có lưu buffer) hoặc 'test' (không học, không lưu)
    """
    state_dict = env.reset()
    total_reward = 0
    sfc_completed = 0
    total_requests = 10 # Số request trong 1 episode
    
    # Hiển thị log chi tiết chỉ cho episode đầu tiên của Train và Test
    show_logs = False
    
    if show_logs:
        print(f"\n--- Running {mode.upper()} Episode {episode_idx} (Detailed Log) ---")

    MAX_STEPS_PER_REQUEST = 50 
    for req_idx in range(total_requests):
        sfc_type, sfc_data = traffic_gen.generate_request()
        env.set_current_request(sfc_data)
        request_active = True
        
        # Tracking cho request này
        is_success = False

        step_count = 0 # <--- [NEW] Reset đếm bước
        
        while request_active:
            step_count += 1 # <--- [NEW] Tăng bước
            
            # --- [NEW] SAFETY CHECK: CHẶN VÒNG LẶP VÔ HẠN ---
            if step_count > MAX_STEPS_PER_REQUEST:
                # Nếu quá lâu mà chưa xong -> Coi như thất bại và Drop request
                if mode == 'test' and episode_idx == 0:
                    print(f"  Req {req_idx}: Force Quit (Exceeded {MAX_STEPS_PER_REQUEST} steps)")
                request_active = False
                break

            # --- 1. DC SELECTION ---
            best_dc_idx = -1
            used_dcs = env.placed_vnfs_loc
            
            # Filter Candidates (Anti-Affinity Constraint)
            candidates = [i for i in range(config.NUM_DCS) if i not in used_dcs]
            
            if not candidates:
                request_active = False # Hết DC trống
                break

            # Epsilon-Greedy (Chỉ có tác dụng nếu epsilon > 0)
            if np.random.rand() < epsilon:
                best_dc_idx = np.random.choice(candidates)
            else:
                # VAE / GenAI Selection (Max Value)
                max_val = -float('inf')
                for dc_idx in candidates:
                    dc_state = env.get_dc_state(dc_idx)
                    val = vae.get_dc_score(dc_state)
                    if val > max_val:
                        max_val = val
                        best_dc_idx = dc_idx
            
            # --- 2. DQN ACTION ---
            # Prepare Inputs
            current_state_dict = env._get_state(selected_dc_idx=best_dc_idx)
            dc_state_curr = env.get_dc_state(best_dc_idx)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                inp = [
                    current_state_dict["dc_state_input"][np.newaxis],
                    current_state_dict["dc_sfc_input"][np.newaxis],
                    current_state_dict["global_sfc_input"][np.newaxis]
                ]
                q_values = dqn.predict(inp, verbose=0)
                action = np.argmax(q_values[0])

            # --- 3. EXECUTE ---
            next_state_dict, reward, done, info = env.step(action, best_dc_idx)
            dc_state_next = env.get_dc_state(best_dc_idx)
            
            total_reward += reward
            
            if show_logs:
                 # In log rút gọn để đỡ spam
                 if reward < 0: status = "Penalty"
                 else: status = "Good"
                 print(f"  Req {req_idx}: DC {best_dc_idx} -> Act {action} -> Rew {reward:.1f} ({status})")

            # --- 4. STORE & TRAIN (ONLY IN TRAIN MODE) ---
            if mode == 'train':
                dqn_buff.push(current_state_dict, action, reward, next_state_dict, done)
                if reward > -1.0:
                    vae_buff.add(dc_state_curr, dc_state_next, reward)
                
                # Training Logic (như code trước)
                train_models()

            # Update Loop
            state_dict = next_state_dict
            
            if done:
                # Nếu done=True và reward dương lớn (hoàn thành SFC)
                if reward >= 2.0: 
                    sfc_completed += 1
                    is_success = True
                request_active = False
                
    return total_reward, sfc_completed

def train_models():
    """Hàm phụ trợ để gọi training step"""
    # 1. Train DQN
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
        
        dqn.fit([s_in1, s_in2, s_in3], curr_q, epochs=1, verbose=0)

    # 2. Train VAE
    if len(vae_buff.current_states) > config.BATCH_SIZE:
        curr_b, next_b, val_b = vae_buff.get_batch(config.BATCH_SIZE)
        # Convert to tensor
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

def main():
    # --- PHASE 1: TRAINING ---
    NUM_TRAIN_EPISODES = 500 # Để demo nhanh, thực tế nên 500+
    epsilon = 1.0
    epsilon_decay = 0.95
    min_epsilon = 0.05
    
    train_rewards = []
    
    print("==========================================")
    print(f"STARTING TRAINING ({NUM_TRAIN_EPISODES} Episodes)")
    print("==========================================")
    
    pbar = trange(NUM_TRAIN_EPISODES, desc="Training", unit="ep")
    for ep in pbar:
        # Gọi hàm run_episode với mode='train'
        reward, _ = run_episode(ep, mode='train', epsilon=epsilon)
        train_rewards.append(reward)
        
        # Decay Epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            
        pbar.set_postfix(rew=f"{reward:.1f}", eps=f"{epsilon:.2f}")

    # --- PHASE 2: TESTING (EVALUATION) ---
    NUM_TEST_EPISODES = 50
    test_rewards = []
    success_rates = [] # Acceptance Ratio
    
    print("\n==========================================")
    print(f"STARTING TESTING ({NUM_TEST_EPISODES} Episodes)")
    print("Mode: No Exploration (Epsilon=0), No Learning")
    print("==========================================")
    
    pbar_test = trange(NUM_TEST_EPISODES, desc="Testing", unit="ep")
    for ep in pbar_test:
        # Gọi hàm run_episode với mode='test', epsilon=0
        reward, successes = run_episode(ep, mode='test', epsilon=0.0)
        
        test_rewards.append(reward)
        # Mỗi episode có 10 requests
        acc_ratio = (successes / 10.0) * 100 
        success_rates.append(acc_ratio)
        
        pbar_test.set_postfix(rew=f"{reward:.1f}", acc=f"{acc_ratio:.0f}%")

    avg_test_reward = np.mean(test_rewards)
    avg_acc_ratio = np.mean(success_rates)
    
    print("\n==========================================")
    print("FINAL RESULTS")
    print(f"Average Test Reward: {avg_test_reward:.2f}")
    print(f"Average Acceptance Ratio: {avg_acc_ratio:.2f}%")
    print("==========================================")

    # --- PHASE 3: VISUALIZATION ---
    plot_results(train_rewards, test_rewards, success_rates)

def plot_results(train_rew, test_rew, success_rates):
    plt.figure(figsize=(15, 5))

    # Biểu đồ 1: Quá trình học (Reward Training)
    plt.subplot(1, 3, 1)
    plt.plot(train_rew, label='Training Reward', color='blue')
    plt.title('Training Convergence')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()

    # Biểu đồ 2: Reward khi Test (để xem độ ổn định)
    plt.subplot(1, 3, 2)
    plt.plot(test_rew, label='Test Reward', color='green')
    plt.title('Evaluation Performance')
    plt.xlabel('Test Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()

    # Biểu đồ 3: Acceptance Ratio (Quan trọng nhất theo bài báo)
    plt.subplot(1, 3, 3)
    plt.plot(success_rates, label='Acceptance Ratio (%)', color='red', marker='o')
    plt.axhline(y=np.mean(success_rates), color='black', linestyle='--', label='Average')
    plt.title('SFC Acceptance Ratio')
    plt.xlabel('Test Episode')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105) # 0-100%
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()