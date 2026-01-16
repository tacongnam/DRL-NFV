import os
import numpy as np
import config
from envs import RandomSelector, Observer, PrioritySelector
from agents import VAETrainer, DQNAgent

SAMPLING_INTERVAL = 1 

def collect_and_train_vae_random(runner, num_episodes, dc_selector, dqn_model_path='models/best_model', vae_epochs=None):
    from runners.data_generator import DataGenerator
    
    if dc_selector is None or isinstance(dc_selector, RandomSelector):
        dc_selector = PrioritySelector()
    
    # Fallback to config if not provided
    if vae_epochs is None:
        vae_epochs = config.GENAI_VAE_EPOCHS

    print(f"\n{'='*80}", flush=True)
    print(f"Collecting VAE Data (Optimized Speed)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    trainer = None
    dqn_agent = None
    
    for ep_idx in range(num_episodes):
        try:
            nodes_range = (20, 35)
            req_range = (50, 80)
            
            scenario_data = DataGenerator.generate_scenario(
                num_nodes_range=nodes_range, server_ratio=0.2,
                num_vnf_types=config.MAX_VNF_TYPES, num_requests_range=req_range
            )
            runner.load_from_dict(scenario_data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            if trainer is None:
                first_server = next((d for d in runner.dcs if d.is_server), None)
                dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None, env.topology)
                trainer = VAETrainer(dummy_state.shape[0])
            
            if dqn_agent is None and os.path.exists(f"{dqn_model_path}_q.weights.h5"):
                state_shapes = [s.shape for s in env.observation_space.spaces]
                dqn_agent = DQNAgent(state_shapes, env.action_space.n)
                dqn_agent.load(dqn_model_path)
                
            state, _ = env.reset()
            done = False
            
            while not done:
                step_idx = env.step_count
                should_collect = len(env.sfc_manager.active_requests) > 0
                
                prev_vae_states = {}
                if should_collect:
                    active_reqs = Observer.get_active_requests(env.sfc_manager)
                    global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
                    for dc in env.dcs:
                        if dc.is_server:
                            prev_vae_states[dc.id] = Observer.get_dc_state(dc, env.sfc_manager, global_stats, env.topology)
                
                mask = env._get_valid_actions_mask()
                if dqn_agent: 
                    action = dqn_agent.select_action(state, epsilon=0.1, valid_mask=mask)
                else: 
                    valid = np.where(mask)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 0
                
                next_state, _, done, _, _ = env.step(action)
                state = next_state
                
                if should_collect:
                    active_reqs_new = Observer.get_active_requests(env.sfc_manager)
                    global_stats_new = Observer.precompute_global_stats(env.sfc_manager, active_reqs_new)
                    
                    for dc in env.dcs:
                        if dc.is_server:
                            prev_s = prev_vae_states.get(dc.id)
                            if prev_s is not None:
                                curr_s = Observer.get_dc_state(dc, env.sfc_manager, global_stats_new, env.topology)
                                value = Observer.calculate_dc_value(dc, env.sfc_manager, prev_s, global_stats_new)
                                trainer.collect_transition(prev_s, curr_s, value)
            
            if (ep_idx + 1) % 20 == 0:
                print(f"Collected Episodes {ep_idx+1}/{num_episodes} | Total Samples: {trainer.size}", flush=True)

        except Exception as e:
            print(f"Skip episode error: {e}")
            continue
    
    min_samples = config.GENAI_BATCH_SIZE * 2
    if trainer and trainer.size > min_samples:
        print(f"\nTraining VAE on {trainer.size} samples...", flush=True)
        trainer.train_vae(epochs=vae_epochs)
        trainer.train_value_network(epochs=vae_epochs)
        
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/vae_model")
        np.savez("models/vae_model_norm.npz", mean=trainer.agent.value_mean, std=trainer.agent.value_std)
        print("VAE Model saved.")
        return trainer.agent
    else:
        print(f"⚠️ Not enough samples ({trainer.size if trainer else 0}). Need at least {min_samples}.")
        return None
    
def collect_and_train_vae_file(runner, file_path, num_episodes, dc_selector):
    pass