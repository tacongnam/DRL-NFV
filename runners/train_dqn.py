import os
import numpy as np
import config
from agents import DQNAgent

def run_episode(env, agent, epsilon, training=True):
    """Chạy 1 episode. Trả về: (reward, acceptance_ratio, dropped_count, decision_steps)"""
    state, _ = env.reset()
    total_reward = 0
    decision_steps = 0 # Đếm số lần Agent thực sự ra quyết định
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, epsilon, mask)
        next_state, reward, done, _, info = env.step(action)
        
        if training:
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train step: Chỉ train khi Agent đã thu thập đủ dữ liệu
            # Lưu ý: Với Event Skipping, decision_steps tăng chậm, nên ta train mỗi bước
            # hoặc giảm TRAIN_INTERVAL xuống 1
            if len(agent.memory) > config.BATCH_SIZE:
                # Train mỗi bước hoặc mỗi 2 bước để bù lại việc số step bị ít đi
                if decision_steps % 2 == 0: 
                    agent.train()
                
                # Update Target Network
                # Cần update thường xuyên hơn vì số step ít đi
                # Ví dụ: Cứ 50 decision steps update 1 lần thay vì 500
                if decision_steps % 50 == 0: 
                    agent.update_target_network()
        
        state = next_state
        total_reward += reward
        decision_steps += 1
        
    stats = env.sfc_manager.get_statistics()
    return total_reward, stats['acceptance_ratio'], stats['dropped_count'], decision_steps

def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*60}\nTraining DQN ({num_episodes} eps) - Optimized Decay\n{'='*60}")
    
    agent = None
    hist = {'rw': [], 'ar': [], 'drop': []}
    
    # Không dùng total_steps cố định nữa vì Event Skipping làm thay đổi số step thực tế
    
    for ep in range(num_episodes):
        try:
            # Curriculum Learning Config
            p = ep / num_episodes
            # Tăng độ khó dần
            if p < 0.3: cfg = ((20, 30), (30, 50))
            elif p < 0.6: cfg = ((30, 50), (50, 80))
            else: cfg = ((40, 60), (80, 120))
            
            # Setup Env
            data = DataGenerator.generate_scenario(num_nodes_range=cfg[0], server_ratio=0.2, num_requests_range=cfg[1])
            runner.load_from_dict(data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            # Init Agent (Lazy loading)
            if not agent:
                agent = DQNAgent([s.shape for s in env.observation_space.spaces], env.action_space.n)
            
            # --- FIX: EPSILON DECAY THEO EPISODE ---
            # Giảm epsilon dựa trên % số episode đã chạy
            # Công thức này đảm bảo Epsilon giảm từ 1.0 xuống ~0.01 ở khoảng 85% quá trình train
            decay_factor = 4.0 # Hệ số mũ, càng cao giảm càng nhanh
            epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * np.exp(-decay_factor * p)
            
            # Run
            rw, ar, drop, steps = run_episode(env, agent, epsilon)
            
            # Log
            hist['rw'].append(rw); hist['ar'].append(ar); hist['drop'].append(drop)
            
            if (ep + 1) % 50 == 0:
                avg_rw = np.mean(hist['rw'])
                avg_ar = np.mean(hist['ar'])
                avg_drop = np.mean(hist['drop'])
                print(f"Ep {ep+1}/{num_episodes} | Rw: {avg_rw:.1f} | AR: {avg_ar:.1f}% | Drop: {avg_drop:.1f} | DecSteps: {steps} | Eps: {epsilon:.3f}")
                hist = {'rw': [], 'ar': [], 'drop': []} # Reset buffer
            
            if (ep + 1) % 100 == 0:
                agent.save(f"models/checkpoint_{ep+1}")
                
        except Exception as e:
            print(f"Skip Ep {ep}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        print("Saved models/best_model")
    
    return agent