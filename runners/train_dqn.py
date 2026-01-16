import os
import numpy as np
import config
from agents import DQNAgent

def run_episode(env, agent, epsilon, training=True):
    """Chạy 1 episode. Trả về: (reward, acceptance_ratio, dropped_count, decision_steps)"""
    state, _ = env.reset()
    total_reward = 0
    decision_steps = 0 
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, epsilon, mask)
        next_state, reward, done, _, info = env.step(action)
        
        if training:
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train step: 
            # Với Event Skipping, ta train tích cực hơn: mỗi 4 bước ra quyết định train 1 lần
            if len(agent.memory) > config.BATCH_SIZE:
                if decision_steps % 4 == 0: 
                    agent.train()
                
                # Update Target Network: mỗi 100 quyết định
                if decision_steps % 100 == 0: 
                    agent.update_target_network()
        
        state = next_state
        total_reward += reward
        decision_steps += 1
        
    stats = env.sfc_manager.get_statistics()
    return total_reward, stats['acceptance_ratio'], stats['dropped_count'], decision_steps

def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*60}\nTraining DQN ({num_episodes} eps) - Linear Decay\n{'='*60}")
    
    agent = None
    hist = {'rw': [], 'ar': [], 'drop': []}
    
    # Cấu hình Linear Decay
    # Epsilon sẽ giảm từ 1.0 xuống 0.01 trong 80% số episode đầu (vd: ep 0 -> 400)
    # 20% episode cuối (ep 401 -> 500) sẽ giữ nguyên 0.01 để ổn định
    exploration_fraction = 0.8 
    decay_duration = int(num_episodes * exploration_fraction)
    
    for ep in range(num_episodes):
        try:
            # Curriculum Learning: Tăng độ khó theo tiến độ
            p = ep / num_episodes
            if p < 0.3: cfg = ((20, 30), (30, 50))      # Easy
            elif p < 0.6: cfg = ((30, 50), (50, 80))    # Medium
            else: cfg = ((40, 60), (80, 120))           # Hard
            
            # Setup Env
            data = DataGenerator.generate_scenario(num_nodes_range=cfg[0], server_ratio=0.2, num_requests_range=cfg[1])
            runner.load_from_dict(data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            # Init Agent
            if not agent:
                agent = DQNAgent([s.shape for s in env.observation_space.spaces], env.action_space.n)
            
            # --- NEW: LINEAR DECAY FORMULA ---
            if ep < decay_duration:
                # Giảm đều tuyến tính
                progress = ep / decay_duration
                epsilon = config.EPSILON_START - progress * (config.EPSILON_START - config.EPSILON_MIN)
            else:
                # Cố định mức thấp nhất
                epsilon = config.EPSILON_MIN
            
            # Run
            rw, ar, drop, steps = run_episode(env, agent, epsilon)
            
            # Log
            hist['rw'].append(rw); hist['ar'].append(ar); hist['drop'].append(drop)
            
            if (ep + 1) % 20 == 0: # Log thường xuyên hơn
                avg_rw = np.mean(hist['rw'])
                avg_ar = np.mean(hist['ar'])
                avg_drop = np.mean(hist['drop'])
                print(f"Ep {ep+1}/{num_episodes} | Rw: {avg_rw:.1f} | AR: {avg_ar:.1f}% | Steps: {steps} | Eps: {epsilon:.3f}")
                hist = {'rw': [], 'ar': [], 'drop': []} 
            
            if (ep + 1) % 100 == 0:
                agent.save(f"models/checkpoint_{ep+1}")
                
        except Exception as e:
            print(f"Skip Ep {ep}: {e}")
            continue

    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        print("Saved models/best_model")
    
    return agent