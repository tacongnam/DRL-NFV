import os
import numpy as np
import config
from agents import DQNAgent

def train_dqn_on_env(runner, env, agent, num_updates, dc_selector):
    print(f"Starting training: {num_updates} updates", flush=True)
    
    sim_loops_per_ep = config.MAX_SIM_TIME_PER_EPISODE / config.TIME_STEP
    steps_per_episode = sim_loops_per_ep * config.ACTIONS_PER_TIME_STEP
    total_training_steps = num_updates * config.EPISODES_PER_UPDATE * steps_per_episode
    
    rewards, acceptances = [], []
    best_ar = 0.0
    global_step = 0
    
    for update in range(1, num_updates + 1):
        print(f"\nUpdate {update}/{num_updates}", flush=True)
        update_rewards, update_ars = [], []
        
        for ep in range(config.EPISODES_PER_UPDATE):
            print(f"  Episode {ep+1}/{config.EPISODES_PER_UPDATE}...", end='', flush=True)
            state, _ = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            
            while not done:
                epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                          np.exp(-1.0 * global_step * 3 / total_training_steps)
                
                mask = env._get_valid_actions_mask()
                action = agent.select_action(state, epsilon, mask)
                next_state, reward, done, _, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                
                if env.step_count % config.TRAIN_INTERVAL == 0:
                    agent.train()
                if env.step_count % config.TARGET_NETWORK_UPDATE == 0:
                    agent.update_target_network()
                
                state = next_state
                ep_reward += reward
                global_step += 1
                step_count += 1
            
            update_rewards.append(ep_reward)
            update_ars.append(info['acceptance_ratio'])
            print(f" R={ep_reward:.1f} AR={info['acceptance_ratio']:.1f}%", flush=True)
        
        avg_ar = np.mean(update_ars)
        avg_r = np.mean(update_rewards)
        rewards.extend(update_rewards)
        acceptances.extend(update_ars)
        
        print(f"Update {update} Summary: Avg AR={avg_ar:.2f}% Avg R={avg_r:.1f}", flush=True)
        
        if avg_ar > best_ar:
            best_ar = avg_ar
            os.makedirs("models", exist_ok=True)
            agent.save("models/best_model")
            print(f"  ★ Best model saved: {best_ar:.2f}%", flush=True)
    
    print(f"\nTraining complete!", flush=True)
    return agent, rewards, acceptances

def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*80}", flush=True)
    print(f"Training DQN with Random Data Generation", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"MAX_VNF_TYPES: {config.MAX_VNF_TYPES}", flush=True)
    print(f"Action Space: {config.ACTION_SPACE_SIZE}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    agent = None
    all_rewards, all_acceptances = [], []
    
    total_training_steps = num_episodes * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP / config.TIME_STEP
    global_step = 0
    
    for ep_idx in range(num_episodes):
        try:
            num_vnf_types = np.random.randint(4, config.MAX_VNF_TYPES + 1)
            
            scenario_data = DataGenerator.generate_scenario(
                num_dcs_range=(5, 10),
                num_switches_range=(30, 70),
                num_vnf_types=num_vnf_types,
                num_requests_range=(50, 80)
            )
            
            num_dcs = len([v for v in scenario_data['V'].values() if v.get('server', False)])
            num_switches = len([v for v in scenario_data['V'].values() if not v.get('server', False)])
            num_requests = len(scenario_data['R'])
            num_edges = len(scenario_data['E'])
            
            if (ep_idx + 1) % 10 == 0:
                print(f"\n{'='*60}", flush=True)
                print(f"Episode {ep_idx+1}/{num_episodes}", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"Scenario Config:", flush=True)
                print(f"  DCs: {num_dcs} | Switches: {num_switches} | VNF Types: {num_vnf_types}", flush=True)
                print(f"  Requests: {num_requests} | Links: {num_edges}", flush=True)
            else:
                print(f"Ep {ep_idx+1}/{num_episodes} [DC:{num_dcs} SW:{num_switches} VNF:{num_vnf_types} Req:{num_requests}]: ", end='', flush=True)
            
            runner.load_from_dict(scenario_data)
            env = runner.create_env(dc_selector)
            
        except Exception as e:
            print(f"ERROR generating scenario: {e}", flush=True)
            continue
        
        if hasattr(env.observation_space, 'spaces'):
            state_shapes = [s.shape for s in env.observation_space.spaces]
        else:
            state_shapes = [env.observation_space.shape]
        
        if agent is None:
            agent = DQNAgent(state_shapes, env.action_space.n)
            print(f"\nAgent created:", flush=True)
            print(f"  State shapes: {state_shapes}", flush=True)
            print(f"  Action space: {env.action_space.n}", flush=True)
            print(f"  Memory size: {config.MEMORY_SIZE}\n", flush=True)
        
        try:
            state, _ = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            
            while not done:
                epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                          np.exp(-1.0 * global_step * 3 / total_training_steps)
                
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
            
            all_rewards.append(ep_reward)
            all_acceptances.append(info['acceptance_ratio'])
            
            stats = env.sfc_manager.get_statistics()
            completed = stats['completed_count']
            dropped = stats['dropped_count']
            
            if (ep_idx + 1) % 10 != 0:
                print(f"R={ep_reward:.0f} AR={info['acceptance_ratio']:.1f}% (C:{completed} D:{dropped} S:{step_count})", flush=True)
            else:
                print(f"Results:", flush=True)
                print(f"  Reward: {ep_reward:.0f}", flush=True)
                print(f"  Acceptance Ratio: {info['acceptance_ratio']:.1f}%", flush=True)
                print(f"  Completed: {completed} | Dropped: {dropped}", flush=True)
                print(f"  Steps: {step_count} | Epsilon: {epsilon:.4f}", flush=True)
            
        except Exception as e:
            print(f"ERROR in episode: {e}", flush=True)
            import traceback
            traceback.print_exc()
        
        if (ep_idx + 1) % 50 == 0:
            recent_ar = np.mean(all_acceptances[-50:])
            recent_r = np.mean(all_rewards[-50:])
            recent_completed = np.mean([env.sfc_manager.get_statistics()['completed_count'] for _ in range(min(50, len(all_acceptances)))])
            
            print(f"\n{'*'*60}", flush=True)
            print(f"Checkpoint {ep_idx+1}/{num_episodes}", flush=True)
            print(f"{'*'*60}", flush=True)
            print(f"Last 50 Episodes Statistics:", flush=True)
            print(f"  Avg Acceptance Ratio: {recent_ar:.2f}%", flush=True)
            print(f"  Avg Reward: {recent_r:.1f}", flush=True)
            print(f"  Memory Size: {len(agent.memory)}/{config.MEMORY_SIZE}", flush=True)
            print(f"{'*'*60}\n", flush=True)
            
            if agent:
                os.makedirs("models", exist_ok=True)
                agent.save(f"models/checkpoint_{ep_idx+1}")
                print(f"Model checkpoint saved\n", flush=True)
    
    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        final_ar = np.mean(all_acceptances[-100:]) if len(all_acceptances) >= 100 else np.mean(all_acceptances)
        final_r = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        
        print(f"\n{'='*80}", flush=True)
        print(f"TRAINING COMPLETE", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total Episodes: {num_episodes}", flush=True)
        print(f"Final Performance (last 100 episodes):", flush=True)
        print(f"  Acceptance Ratio: {final_ar:.2f}%", flush=True)
        print(f"  Average Reward: {final_r:.1f}", flush=True)
        print(f"  Memory Utilized: {len(agent.memory)}/{config.MEMORY_SIZE}", flush=True)
        print(f"Models Saved:", flush=True)
        print(f"  - models/best_model", flush=True)
        print(f"  - models/checkpoint_* (every 50 episodes)", flush=True)
        print(f"{'='*80}\n", flush=True)
    
    return agent, all_rewards, all_acceptances