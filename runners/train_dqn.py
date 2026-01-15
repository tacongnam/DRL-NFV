import os
import numpy as np
import traceback
import config
from agents import DQNAgent

def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*80}", flush=True)
    print(f"Training DQN with Scale-Free Network Generation", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    agent = None
    log_rewards, log_ars, log_steps, log_dropped = [], [], [], []
    
    # Tính tổng step để decay epsilon
    total_training_steps = num_episodes * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP / config.TIME_STEP
    global_step = 0
    
    for ep_idx in range(num_episodes):
        try:
            # Curriculum Learning đơn giản
            progress = ep_idx / num_episodes
            if progress < 0.3:
                nodes_range, req_range = (20, 30), (30, 50) # Easy
            elif progress < 0.6:
                nodes_range, req_range = (30, 50), (50, 80) # Medium
            else:
                nodes_range, req_range = (40, 60), (80, 120) # Hard
            
            # Generate Data
            scenario_data = DataGenerator.generate_scenario(
                num_nodes_range=nodes_range, server_ratio=0.2,
                num_vnf_types=config.MAX_VNF_TYPES, num_requests_range=req_range
            )
            
            # Load Data & Init Env
            runner.load_from_dict(scenario_data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            # Init Agent (Lazy loading)
            if agent is None:
                state_shapes = [s.shape for s in env.observation_space.spaces]
                agent = DQNAgent(state_shapes, env.action_space.n)
                print(f"Initialized Agent. Action size: {env.action_space.n}", flush=True)
            
            # Episode Loop
            state, _ = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            
            while not done:
                # Epsilon Decay
                epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * np.exp(- global_step * 4 / total_training_steps)
                
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
            
            # Stats
            stats = env.sfc_manager.get_statistics()
            log_rewards.append(ep_reward)
            log_ars.append(info['acceptance_ratio'])
            log_steps.append(step_count)
            log_dropped.append(stats['dropped_count'])
            
            # Logging
            if (ep_idx + 1) % 10 == 0:
                print(f"Ep {ep_idx+1}/{num_episodes} | "
                      f"Rw: {np.mean(log_rewards):.1f} | AR: {np.mean(log_ars):.1f}% | "
                      f"Drop: {np.mean(log_dropped):.1f} | Eps: {epsilon:.3f}", flush=True)
                log_rewards, log_ars, log_steps, log_dropped = [], [], [], []
            
            # Save Checkpoint
            if (ep_idx + 1) % 50 == 0:
                os.makedirs("models", exist_ok=True)
                agent.save(f"models/checkpoint_{ep_idx+1}")

        except Exception as e:
            # QUAN TRỌNG: In lỗi ra thay vì im lặng
            print(f"Error in Episode {ep_idx}: {e}")
            traceback.print_exc()
            continue

    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        print("\nTraining Complete. Model saved to models/best_model")
    else:
        print("\nTraining Failed: Agent was never initialized.")
    
    return agent, [], []