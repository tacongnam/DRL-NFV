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
    
    print(f"\n{'='*80}", flush=True)
    print(f"Collecting VAE Data (Scale-Free Topology)", flush=True)
    print(f"Total Episodes: {num_episodes}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    trainer = None
    dqn_agent = None
    
    for ep_idx in range(num_episodes):
        try:
            # Sinh kịch bản đa dạng độ khó...
            if ep_idx < num_episodes * 0.3: nodes_range, req_range = (30, 40), (50, 80)
            elif ep_idx < num_episodes * 0.7: nodes_range, req_range = (40, 60), (80, 150)
            else: nodes_range, req_range = (60, 90), (150, 250)
            
            scenario_data = DataGenerator.generate_scenario(
                num_nodes_range=nodes_range, server_ratio=0.15,
                num_vnf_types=config.MAX_VNF_TYPES, num_requests_range=req_range
            )
            runner.load_from_dict(scenario_data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            if trainer is None:
                first_server = next((d for d in runner.dcs if d.is_server), None)
                dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None)
                trainer = VAETrainer(dummy_state.shape[0])
            
            if dqn_agent is None and os.path.exists(f"{dqn_model_path}_q.weights.h5"):
                state_shapes = [s.shape for s in env.observation_space.spaces] if hasattr(env.observation_space, 'spaces') else [env.observation_space.shape]
                dqn_agent = DQNAgent(state_shapes, env.action_space.n)
                dqn_agent.load(dqn_model_path)
                
            state, _ = env.reset()
            done = False
            
            while not done:
                prev_states = {dc.id: Observer.get_dc_state(dc, env.sfc_manager, None) for dc in env.dcs if dc.is_server}
                mask = env._get_valid_actions_mask()
                
                if dqn_agent: action = dqn_agent.select_action(state, epsilon=0.1, valid_mask=mask)
                else: action = np.random.choice(np.where(mask)[0]) if len(np.where(mask)[0]) > 0 else 0
                
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
            
            # LOGGING MỖI 10 EPISODES
            if (ep_idx + 1) % 10 == 0:
                print(f"Collected Episodes {ep_idx-8}-{ep_idx+1}/{num_episodes} | Total Samples: {trainer.size}", flush=True)

        except Exception as e:
            continue
    
    if trainer and trainer.size > 1000:
        print(f"\nTraining VAE on {trainer.size} samples...", flush=True)
        trainer.train_vae()
        trainer.train_value_network()
        
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/vae_model")
        np.savez("models/vae_model_norm.npz", mean=trainer.agent.value_mean, std=trainer.agent.value_std)
        print("VAE Model saved.")
        return trainer.agent
    return None