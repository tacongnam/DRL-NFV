import os
import numpy as np
import config
from envs import RandomSelector, Observer, PrioritySelector
from agents import VAETrainer, DQNAgent

# Tối ưu: Chỉ thu thập dữ liệu mỗi 10 bước để giảm tải tính toán
SAMPLING_INTERVAL = 10 

def collect_and_train_vae_file(runner, file_path, num_episodes, dc_selector):
    print(f"\nCollecting VAE data from file: {file_path}", flush=True)
    print(f"Episodes: {num_episodes} | Sampling Interval: {SAMPLING_INTERVAL}", flush=True)
    
    runner.load_from(file_path)
    env = runner.create_env(dc_selector)
    
    first_server = next((d for d in runner.dcs if d.is_server), None)
    dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None, None)
    dc_state_dim = dummy_state.shape[0]
    
    trainer = VAETrainer(dc_state_dim)
    
    for ep in range(num_episodes):
        env.reset()
        done = False
        
        while not done:
            step_idx = env.step_count
            should_collect = (step_idx % SAMPLING_INTERVAL == 0)
            
            # 1. Nếu đến lượt collect, chụp trạng thái TRƯỚC khi hành động
            prev_vae_states = {}
            if should_collect:
                # Precompute global stats một lần để dùng cho tất cả DC
                active_reqs = Observer.get_active_requests(env.sfc_manager)
                global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
                
                for dc in env.dcs:
                    if dc.is_server:
                        prev_vae_states[dc.id] = Observer.get_dc_state(dc, env.sfc_manager, global_stats, env.topology)
            
            # 2. Thực hiện hành động (Dùng Random hoặc Selector có sẵn của env)
            mask = env._get_valid_actions_mask()
            valid = np.where(mask)[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0
            
            _, _, done, _, _ = env.step(action)
            
            # 3. Nếu đến lượt collect, chụp trạng thái SAU khi hành động và lưu
            if should_collect:
                # Tính lại stats vì môi trường đã thay đổi sau step
                active_reqs_new = Observer.get_active_requests(env.sfc_manager)
                global_stats_new = Observer.precompute_global_stats(env.sfc_manager, active_reqs_new)
                
                for dc in env.dcs:
                    if dc.is_server:
                        prev_s = prev_vae_states.get(dc.id)
                        if prev_s is not None:
                            curr_s = Observer.get_dc_state(dc, env.sfc_manager, global_stats_new, env.topology)
                            # Tính Value dựa trên trạng thái mới
                            value = Observer.calculate_dc_value(dc, env.sfc_manager, prev_s, global_stats_new)
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
    
    # Nếu không có selector, dùng PrioritySelector để dữ liệu tốt hơn (AR cao hơn)
    if dc_selector is None or isinstance(dc_selector, RandomSelector):
        dc_selector = PrioritySelector()

    print(f"\n{'='*80}", flush=True)
    print(f"Collecting VAE Data (Optimized Speed)", flush=True)
    print(f"Total Episodes: {num_episodes} | Sampling Interval: {SAMPLING_INTERVAL}", flush=True)
    print(f"Selector: {type(dc_selector).__name__}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    trainer = None
    dqn_agent = None
    
    for ep_idx in range(num_episodes):
        try:
            # TỐI ƯU: Sử dụng mạng NHỎ (20-35 nodes) để chạy nhanh
            # VAE học quan hệ cục bộ (Local Resource -> Value), không cần mạng lớn
            nodes_range = (20, 35)
            req_range = (50, 80)
            
            scenario_data = DataGenerator.generate_scenario(
                num_nodes_range=nodes_range, 
                server_ratio=0.2, # Tăng tỷ lệ server một chút để dễ thở
                num_vnf_types=config.MAX_VNF_TYPES, 
                num_requests_range=req_range
            )
            runner.load_from_dict(scenario_data)
            config.update_resource_constraints(runner.dcs, runner.graph)
            env = runner.create_env(dc_selector)
            
            # Init Trainer & Agent nếu chưa có
            if trainer is None:
                first_server = next((d for d in runner.dcs if d.is_server), None)
                dummy_state = Observer.get_dc_state(first_server, env.sfc_manager, None, None)
                trainer = VAETrainer(dummy_state.shape[0])
            
            if dqn_agent is None and os.path.exists(f"{dqn_model_path}_q.weights.h5"):
                state_shapes = [s.shape for s in env.observation_space.spaces] if hasattr(env.observation_space, 'spaces') else [env.observation_space.shape]
                dqn_agent = DQNAgent(state_shapes, env.action_space.n)
                dqn_agent.load(dqn_model_path)
                
            state, _ = env.reset()
            done = False
            
            while not done:
                step_idx = env.step_count
                should_collect = (step_idx % SAMPLING_INTERVAL == 0)
                
                # 1. Chụp trạng thái trước (cho VAE)
                prev_vae_states = {}
                if should_collect:
                    active_reqs = Observer.get_active_requests(env.sfc_manager)
                    global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
                    for dc in env.dcs:
                        if dc.is_server:
                            prev_vae_states[dc.id] = Observer.get_dc_state(dc, env.sfc_manager, global_stats, env.topology)
                
                # 2. DRL Agent chọn hành động (cần chạy mỗi bước)
                mask = env._get_valid_actions_mask()
                if dqn_agent: 
                    action = dqn_agent.select_action(state, epsilon=0.1, valid_mask=mask)
                else: 
                    action = np.random.choice(np.where(mask)[0]) if len(np.where(mask)[0]) > 0 else 0
                
                next_state, _, done, _, _ = env.step(action)
                state = next_state
                
                # 3. Chụp trạng thái sau (cho VAE) và lưu
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
            
            # LOGGING
            if (ep_idx + 1) % 10 == 0:
                print(f"Collected Episodes {ep_idx+1}/{num_episodes} | Total Samples: {trainer.size}", flush=True)

        except Exception as e:
            # print(f"Skip episode error: {e}")
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
    else:
        print("Not enough samples collected to train VAE.")
        return None