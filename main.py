import numpy as np
from config import *
from env.core_network import CoreNetworkEnv
from models.vae_model import GenAIAgent
from models.dqn_model import DQNAgent
from utils.buffers import VAEBuffer, DQNBuffer

def main():
    env = CoreNetworkEnv()
    vae = GenAIAgent()
    dqn = DQNAgent()
    
    vae_buf = VAEBuffer()
    dqn_buf = DQNBuffer()
    
    print("=== Phase 1: Data Collection for GenAI (Random Policy) ===")
    obs = env.reset()
    for _ in range(500):
        # Randomly select a DC
        dc_idx = np.random.randint(NUM_DCS)
        
        # Get Pre-Action State
        dc_state_pre = env._get_dc_states()[dc_idx]
        
        # Random Action
        action = np.random.randint(ACTION_SPACE)
        
        # Step
        next_dc_states, reward, done, _ = env.step(dc_idx, action)
        
        # Store for VAE: (State, NextState, Value=Reward)
        vae_buf.push(dc_state_pre, next_dc_states[dc_idx], [reward])
        
    print(f"Collected {len(vae_buf)} samples. Training VAE...")
    for i in range(200):
        s, ns, v = vae_buf.sample(BATCH_SIZE)
        loss = vae.train_step(s, ns, v)
        if i % 50 == 0: print(f"VAE Loss: {loss:.4f}")

    print("\n=== Phase 2: GenAI-Assisted DRL Training ===")
    rewards = []
    
    for episode in range(200):
        env.reset()
        total_reward = 0
        steps = 0
        
        # Episode length limited by requests logic or steps
        while steps < 50: 
            # 1. GenAI Selects DC
            all_dc_states = env._get_dc_states()
            
            # Epsilon-Greedy for DC selection to allow exploration beyond VAE's initial bias
            if np.random.rand() < dqn.epsilon:
                selected_dc_idx = np.random.randint(NUM_DCS)
            else:
                values = vae.predict_value(all_dc_states)
                selected_dc_idx = np.argmax(values)
            
            # 2. Get Input for DQN based on selected DC
            dc_s, loc_s, glob_s = env.get_dqn_state(selected_dc_idx)
            
            # 3. DQN Selects Action
            action = dqn.act(dc_s, loc_s, glob_s)
            
            # 4. Execute
            next_all_states, reward, done, _ = env.step(selected_dc_idx, action)
            
            # 5. Get Next States for DQN
            # Note: naive approach assumes next DC is same for training tuple
            n_dc_s, _, _ = env.get_dqn_state(selected_dc_idx) 
            # Recalculate SFC states (Local/Global might change if req completed)
            req = env.current_request
            target = VNFS.index(req['chain'][req['current_idx']]) if req['active'] else 0
            n_loc_s = np.array([target/NUM_VNF_TYPES, req['bw']/100, 0, 0]) 
            n_glob_s = glob_s # Simplify for buffer
            
            dqn_buf.push(dc_s, loc_s, glob_s, action, reward, n_dc_s, n_loc_s, n_glob_s, done)
            
            # 6. Train DQN
            if len(dqn_buf) > BATCH_SIZE:
                dqn.train(dqn_buf.sample(BATCH_SIZE))
            
            # 7. Update VAE buffer online
            vae_buf.push(dc_s, next_all_states[selected_dc_idx], [reward])
            if steps % 10 == 0:
                s, ns, v = vae_buf.sample(BATCH_SIZE)
                vae.train_step(s, ns, v)

            total_reward += reward
            steps += 1
            
        dqn.update_target()
        
        acc_ratio = env.total_accepted / max(1, env.total_requests)
        if episode % 10 == 0:
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Acc Rate: {acc_ratio:.2%} | Eps: {dqn.epsilon:.2f}")

if __name__ == "__main__":
    main()