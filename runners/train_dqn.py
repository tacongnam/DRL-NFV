import os
import numpy as np
import config
from agents import DQNAgent

def run_episode(env, agent, epsilon, training=True):
    """Chạy 1 episode. Trả về: (reward, acceptance_ratio, dropped_count, steps)"""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.select_action(state, epsilon, mask)
        next_state, reward, done, _, info = env.step(action)
        
        if training:
            agent.store_transition(state, action, reward, next_state, done)
            # Train step
            if steps > 0:
                if steps % config.TRAIN_INTERVAL == 0: agent.train()
                if steps % config.TARGET_NETWORK_UPDATE == 0: agent.update_target_network()
        
        state = next_state
        total_reward += reward
        steps += 1
        
    stats = env.sfc_manager.get_statistics()
    return total_reward, stats['acceptance_ratio'], stats['dropped_count'], steps

def train_dqn_random(runner, num_episodes, dc_selector):
    from runners.data_generator import DataGenerator
    
    print(f"\n{'='*60}\nTraining DQN ({num_episodes} eps)\n{'='*60}")
    
    agent = None
    # Buffers for logging
    hist = {'rw': [], 'ar': [], 'drop': []}
    total_steps = num_episodes * config.MAX_SIM_TIME_PER_EPISODE * config.ACTIONS_PER_TIME_STEP * config.TIME_STEP
    curr_step = 0
    
    for ep in range(num_episodes):
        try:
            # Curriculum Learning Config
            p = ep / num_episodes
            if p < 0.3: cfg = ((20, 30), (30, 50))
            elif p < 0.6: cfg = ((30, 50), (50, 80))
            else: cfg = ((40, 60), (80, 120))
            
            # Setup Env
            data = DataGenerator.generate_scenario(num_nodes_range=cfg[0], server_ratio=0.2, num_requests_range=cfg[1])
            runner.load_from_dict(data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            # Init Agent
            if not agent:
                agent = DQNAgent([s.shape for s in env.observation_space.spaces], env.action_space.n)
            
            # Decay Epsilon
            epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * np.exp(-3 * curr_step / total_steps)
            
            # Run
            rw, ar, drop, steps = run_episode(env, agent, epsilon)
            curr_step += steps
            
            # Log
            hist['rw'].append(rw); hist['ar'].append(ar); hist['drop'].append(drop)
            
            if (ep + 1) % 25 == 0:
                print(f"Ep {ep+1}/{num_episodes} | Rw: {np.mean(hist['rw']):.1f} | AR: {np.mean(hist['ar']):.1f}% | Drop: {np.mean(hist['drop']):.1f} | Eps: {epsilon:.3f}")
                hist = {'rw': [], 'ar': [], 'drop': []} # Reset buffer
            
            if (ep + 1) % 50 == 0:
                agent.save(f"models/checkpoint_{ep+1}")
                
        except Exception as e:
            print(f"Skip Ep {ep}: {e}")
            continue

    if agent:
        os.makedirs("models", exist_ok=True)
        agent.save("models/best_model")
        print("Saved models/best_model")
    
    return agent