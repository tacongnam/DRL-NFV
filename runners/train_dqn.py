import os
import numpy as np
import config
from agents import DQNAgent

def train_dqn_on_env(runner, env, agent, num_episodes, dc_selector):
    print(f"\n{'='*80}", flush=True)
    print(f"Training DQN on Static Environment", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    total_training_steps = num_episodes * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP / config.TIME_STEP
    global_step = 0
    
    all_rewards, all_acceptances = [], []
    
    # Biến tạm để gộp log mỗi 10 episodes
    log_rewards, log_ars, log_steps = [], [], []
    best_ar = 0.0

    for ep_idx in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        
        while not done:
            progress = global_step / total_training_steps
            if progress < 0.3:
                epsilon = 0.9 - 0.3 * progress / 0.3
            elif progress < 0.7:
                epsilon = 0.6 - 0.4 * (progress - 0.3) / 0.4
            else:
                epsilon = 0.2 - 0.15 * (progress - 0.7) / 0.3
            epsilon = max(epsilon, config.EPSILON_MIN)
            
            mask = env._get_valid_actions_mask()
            action = agent.select_action(state, epsilon, mask)
            next_state, reward, done, _, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            if step_count % config.TRAIN_INTERVAL == 0 and step_count > 0:
                agent.train()
            if step_count % config.TARGET_NETWORK_UPDATE == 0 and step_count > 0:
                agent.update_target_network()
            
            state = next_state
            ep_reward += reward
            global_step += 1
            step_count += 1
        
        all_rewards.append(ep_reward)
        all_acceptances.append(info['acceptance_ratio'])
        
        log_rewards.append(ep_reward)
        log_ars.append(info['acceptance_ratio'])
        log_steps.append(step_count)
        
        # LOGGING MỖI 10 EPISODES
        if (ep_idx + 1) % 10 == 0:
            avg_r = np.mean(log_rewards)
            avg_ar = np.mean(log_ars)
            avg_steps = np.mean(log_steps)
            print(f"Episodes {ep_idx-8}-{ep_idx+1}/{num_episodes} | "
                  f"Avg Reward: {avg_r:.1f} | Avg AR: {avg_ar:.1f}% | "
                  f"Avg Steps: {avg_steps:.0f} | Eps: {epsilon:.3f}", flush=True)
            
            # Lưu model tốt nhất
            if avg_ar > best_ar:
                best_ar = avg_ar
                os.makedirs("models", exist_ok=True)
                agent.save("models/best_model")
            
            log_rewards, log_ars, log_steps = [], [], []

    print("\nTraining complete!", flush=True)
    return agent, all_rewards, all_acceptances


def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*80}", flush=True)
    print(f"Training DQN with Scale-Free Network Generation", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    agent = None
    all_rewards, all_acceptances = [], []
    
    # Biến tạm để gộp log
    log_rewards, log_ars, log_steps, log_dropped = [], [], [], []
    
    total_training_steps = num_episodes * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP / config.TIME_STEP
    global_step = 0
    
    for ep_idx in range(num_episodes):
        try:
            # Curriculum Learning
            progress = ep_idx / num_episodes
            if progress < 0.2:
                nodes_range, req_range, vnf_types = (20, 30), (30, 50), 5
            elif progress < 0.5:
                nodes_range, req_range, vnf_types = (30, 50), (50, 100), 8
            elif progress < 0.8:
                nodes_range, req_range, vnf_types = (50, 80), (100, 200), config.MAX_VNF_TYPES
            else:
                nodes_range, req_range, vnf_types = (80, 100), (200, 300), config.MAX_VNF_TYPES
            
            scenario_data = DataGenerator.generate_scenario(
                num_nodes_range=nodes_range, server_ratio=0.15,
                num_vnf_types=vnf_types, num_requests_range=req_range
            )
            runner.load_from_dict(scenario_data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
        except Exception as e:
            continue
        
        if agent is None:
            state_shapes = [s.shape for s in env.observation_space.spaces] if hasattr(env.observation_space, 'spaces') else [env.observation_space.shape]
            agent = DQNAgent(state_shapes, env.action_space.n)
        
        try:
            state, _ = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            
            while not done:
                progress_step = global_step / total_training_steps
                epsilon = max(0.9 - 0.8 * progress_step, config.EPSILON_MIN)
                
                mask = env._get_valid_actions_mask()
                action = agent.select_action(state, epsilon, mask)
                next_state, reward, done, _, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                
                if step_count % config.TRAIN_INTERVAL == 0 and step_count > 0:
                    agent.train()
                if step_count % config.TARGET_NETWORK_UPDATE == 0 and step_count > 0:
                    agent.update_target_network()
                
                state = next_state
                ep_reward += reward
                step_count += 1
                global_step += 1
            
            stats = env.sfc_manager.get_statistics()
            
            all_rewards.append(ep_reward)
            all_acceptances.append(info['acceptance_ratio'])
            
            log_rewards.append(ep_reward)
            log_ars.append(info['acceptance_ratio'])
            log_steps.append(step_count)
            log_dropped.append(stats['dropped_count'])
            
            # LOGGING MỖI 10 EPISODES
            if (ep_idx + 1) % 10 == 0:
                print(f"Episodes {ep_idx-8}-{ep_idx+1}/{num_episodes} | "
                      f"Avg Reward: {np.mean(log_rewards):.1f} | "
                      f"Avg AR: {np.mean(log_ars):.1f}% | "
                      f"Avg Dropped: {np.mean(log_dropped):.1f} | Eps: {epsilon:.3f}", flush=True)
                # Reset buffers
                log_rewards, log_ars, log_steps, log_dropped = [], [], [], []
            
        except Exception as e:
            pass
        
        # Checkpoint saving
        if (ep_idx + 1) % 50 == 0 and agent:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/checkpoint_{ep_idx+1}")

    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        print("\nTraining Complete. Model saved to models/best_model")
    
    return agent, all_rewards, all_acceptances