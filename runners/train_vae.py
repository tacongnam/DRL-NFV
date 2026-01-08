import os
import numpy as np
import config
from envs import RandomSelector, Observer
from agents import VAETrainer, DQNAgent

def collect_and_train_vae_file(runner, file_path, num_episodes, dc_selector):
    print(f"\nCollecting VAE data from file: {file_path}", flush=True)
    print(f"Episodes: {num_episodes}\n", flush=True)
    
    runner.load_from(file_path)
    env = runner.create_env(dc_selector)
    
    first_server = next((d for d in runner.dcs if d.is_server), None)
    dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None)
    dc_state_dim = dummy_state.shape[0]
    
    trainer = VAETrainer(dc_state_dim)
    
    for ep in range(num_episodes):
        env.reset()
        done = False
        
        while not done:
            prev_states = {dc.id: Observer.get_dc_state(dc, env.sfc_manager, None) 
                          for dc in env.dcs if dc.is_server}
            
            mask = env._get_valid_actions_mask()
            valid = np.where(mask)[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0
            
            _, _, done, _, _ = env.step(action)
            
            active_reqs = Observer.get_active_requests(env.sfc_manager)
            global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
            
            for dc in env.dcs:
                if dc.is_server:
                    prev_s = prev_states[dc.id]
                    curr_s = Observer.get_dc_state(dc, env.sfc_manager, global_stats)
                    value = Observer.calculate_dc_value(dc, env.sfc_manager, prev_s, global_stats)
                    trainer.collect_transition(prev_s, curr_s, value)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: {trainer.size} samples", flush=True)
    
    print(f"\nCollected {trainer.size} transitions from file", flush=True)
    trainer.train_vae()
    trainer.train_value_network()
    
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/vae_model")
    np.savez("models/vae_model_norm.npz", 
             mean=trainer.agent.value_mean, 
             std=trainer.agent.value_std)
    
    print("✓ VAE model saved\n", flush=True)
    return trainer.agent

def collect_and_train_vae_random(runner, num_episodes, dc_selector, dqn_model_path='models/best_model'):
    from runners.data_generator import DataGenerator
    import random
    
    print(f"\n{'='*80}", flush=True)
    print(f"Collecting VAE Data using Trained DQN Agent", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"DQN Model: {dqn_model_path}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    trainer = None
    dqn_agent = None
    
    for ep_idx in range(num_episodes):
        try:
            progress = ep_idx / num_episodes
            
            if progress < 0.3:
                dc_range = (4, 6)
                sw_range = (6, 10)
                req_range = (15, 30)
            elif progress < 0.6:
                dc_range = (5, 8)
                sw_range = (10, 15)
                req_range = (30, 50)
            else:
                dc_range = (6, 10)
                sw_range = (10, 20)
                req_range = (40, 80)
            
            num_vnf_types = random.randint(6, config.MAX_VNF_TYPES)
            
            scenario_data = DataGenerator.generate_scenario(
                num_dcs_range=dc_range,
                num_switches_range=sw_range,
                num_vnf_types=num_vnf_types,
                num_requests_range=req_range
            )
            
            runner.load_from_dict(scenario_data)
            env = runner.create_env(dc_selector)
            
            if trainer is None:
                first_server = next((d for d in runner.dcs if d.is_server), None)
                dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None)
                dc_state_dim = dummy_state.shape[0]
                trainer = VAETrainer(dc_state_dim)
                print(f"VAE Trainer created with state dim: {dc_state_dim}", flush=True)
            
            if dqn_agent is None:
                if hasattr(env.observation_space, 'spaces'):
                    state_shapes = [s.shape for s in env.observation_space.spaces]
                else:
                    state_shapes = [env.observation_space.shape]
                
                dqn_agent = DQNAgent(state_shapes, env.action_space.n)
                
                q_weights = f"{dqn_model_path}_q.weights.h5"
                target_weights = f"{dqn_model_path}_target.weights.h5"
                
                if os.path.exists(q_weights) and os.path.exists(target_weights):
                    dqn_agent.load(dqn_model_path)
                    print(f"✓ Loaded trained DQN from {dqn_model_path}_*.weights.h5", flush=True)
                else:
                    print(f"⚠ DQN model not found at {dqn_model_path}_*.weights.h5", flush=True)
                    print(f"   Using random agent for data collection", flush=True)
                    dqn_agent = None
                
                print()
            
            if (ep_idx + 1) % 10 == 0:
                print(f"Episode {ep_idx+1}/{num_episodes}: ", end='', flush=True)
            
            state, _ = env.reset()
            done = False
            
            while not done:
                prev_states = {dc.id: Observer.get_dc_state(dc, env.sfc_manager, None) 
                              for dc in env.dcs if dc.is_server}
                
                mask = env._get_valid_actions_mask()
                
                if dqn_agent is not None:
                    action = dqn_agent.select_action(state, epsilon=0.0, valid_mask=mask)
                else:
                    valid = np.where(mask)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 0
                
                next_state, _, done, _, _ = env.step(action)
                state = next_state
                
                active_reqs = Observer.get_active_requests(env.sfc_manager)
                global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
                
                for dc in env.dcs:
                    if dc.is_server:
                        prev_s = prev_states[dc.id]
                        curr_s = Observer.get_dc_state(dc, env.sfc_manager, global_stats)
                        value = Observer.calculate_dc_value(dc, env.sfc_manager, prev_s, global_stats)
                        trainer.collect_transition(prev_s, curr_s, value)
            
            if (ep_idx + 1) % 10 == 0:
                print(f"{trainer.size} samples", flush=True)
                
        except Exception as e:
            print(f"ERROR in episode {ep_idx+1}: {e}", flush=True)
            continue
        
        if (ep_idx + 1) % 50 == 0:
            print(f"\nCheckpoint {ep_idx+1}: Total samples = {trainer.size}", flush=True)
    
    if trainer:
        print(f"\n{'='*80}", flush=True)
        print(f"Training VAE on {trainer.size} samples", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        trainer.train_vae()
        trainer.train_value_network()
        
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/vae_model")
        np.savez("models/vae_model_norm.npz", 
                 mean=trainer.agent.value_mean, 
                 std=trainer.agent.value_std)
        
        print(f"\n{'='*80}", flush=True)
        print(f"VAE Training Complete!", flush=True)
        print(f"  Model saved: models/vae_model", flush=True)
        print(f"  Normalization: models/vae_model_norm.npz", flush=True)
        print(f"  Data collected using: {'Trained DQN' if dqn_agent else 'Random Agent'}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        return trainer.agent
    
    return None