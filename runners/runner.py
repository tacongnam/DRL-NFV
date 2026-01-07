import os
import numpy as np
import config
from envs import SFCEnvironment, RandomSelector, Observer
from agents import DQNAgent, VAETrainer
from runners import load_data

class Runner:
    """Unified training and evaluation pipeline"""
    
    def __init__(self):
        self.graph = None
        self.dc = None
        self.requests = None
        
    @classmethod
    def load_from(self, file_path="test.json"):
        """
        Auto reset before new load!
        """
        Runner.reset()

        graph, dcs, requests, vnf_specs = load_data(file_path=file_path)
        
        config.update_vnf_specs(vnf_specs)
        self.graph = graph
        self.dcs = dcs
        self.requests = requests
        
        print(f"Data loaded: {len([d for d in dcs if d.is_server])} servers, "
              f"{config.NUM_VNF_TYPES} VNFs, {len(requests)} requests")
        return

    @classmethod
    def reset(self):
        self.graph = None
        self.dcs = None
        self.requests = None

    @classmethod
    def create_env(self, dc_selector = None):
        try:
            return SFCEnvironment(self.graph, self.dcs, self.requests, dc_selector=dc_selector)
        except:
            return -1
    
    @classmethod
    def train_dqn(self, dc_selector, save_prefix="", num_updates=None):
        """Train DQN with Exponential Epsilon Decay"""
        if num_updates is None:
            num_updates = config.TRAIN_UPDATES

        env = self.create_env(dc_selector)
        
        # Xử lý shape cho DQN đa đầu vào
        if hasattr(env.observation_space, 'spaces'):
            dqn_state_shapes = [s.shape for s in env.observation_space.spaces]
        else:
            dqn_state_shapes = [env.observation_space.shape]
            
        agent = DQNAgent(dqn_state_shapes, env.action_space.n)
        
        print(f"\n{'='*80}\nTraining DQN ({save_prefix or 'standard'})\n{'='*80}")
        
        # --- TÍNH TOÁN DECAY STEP ---
        sim_loops_per_ep = config.MAX_SIM_TIME_PER_EPISODE / config.TIME_STEP
        steps_per_episode = sim_loops_per_ep * config.ACTIONS_PER_TIME_STEP
        total_training_steps = num_updates * config.EPISODES_PER_UPDATE * steps_per_episode
        
        print(f"Total Training Steps: {int(total_training_steps)}")
        
        rewards, acceptances = [], []
        best_ar = 0.0
        
        # Biến đếm tổng số bước đã thực hiện
        global_step = 0
        
        os.makedirs("models", exist_ok=True)
        
        for update in range(1, num_updates + 1):
            update_rewards, update_ars = [], []
            
            for ep in range(config.EPISODES_PER_UPDATE):
                state, _ = env.reset()
                done = False
                ep_reward = 0
                
                while not done:
                    # --- TÍNH EPSILON THEO CÔNG THỨC ---
                    # epsilon = min + (start - min) * exp(-global_step * 3 / decay_step)
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
                    
                    # Tăng biến đếm global
                    global_step += 1
                
                update_rewards.append(ep_reward)
                update_ars.append(info['acceptance_ratio'])
                
                print(f"  Update {update}/{num_updates} Ep {ep+1}: "
                      f"R={ep_reward:.1f} AR={info['acceptance_ratio']:.1f}% "
                      f"ε={epsilon:.4f}", end="\r", flush=True)
            
            avg_ar = np.mean(update_ars)
            rewards.extend(update_rewards)
            acceptances.extend(update_ars)
            
            print(f"\nUpdate {update}: Avg AR={avg_ar:.2f}%")
            
            if avg_ar > best_ar:
                best_ar = avg_ar
                agent.save(f"models/best_{save_prefix}model")
                print(f"  ★ Best saved: {best_ar:.2f}%")
            
            if update % 50 == 0:
                agent.save(f"models/{save_prefix}checkpoint_{update}")
        
        agent.save(f"models/{save_prefix}final")
        self._plot_training(rewards, acceptances, f"fig/{save_prefix}training.png")
        
        print(f"\n{'='*80}\nTraining complete: Best AR={best_ar:.2f}%\n{'='*80}")
        return agent

    @classmethod
    # ... (Giữ nguyên các hàm collect_vae_data, evaluate, _plot_training, _plot_eval)
    def collect_vae_data(self, num_episodes=None):
        """Collect transitions for VAE training"""
        if num_episodes is None:
            num_episodes = config.GENAI_DATA_EPISODES
        
        env = self.create_env(dc_selector=RandomSelector())
        
        first_server_dc = next((d for d in self.dcs if d.is_server), None)
        if first_server_dc is None:
            raise ValueError("No server DCs found to determine VAE state dimension.")
        
        dummy_dc_state = Observer.get_dc_state(first_server_dc, env.sfc_manager, None)
        dc_state_dim = dummy_dc_state.shape[0]
        
        trainer = VAETrainer(dc_state_dim) 
        
        print(f"\n{'='*80}\nCollecting VAE data: {num_episodes} episodes\n{'='*80}")
        
        for ep in range(num_episodes):
            env.reset()
            done = False
            
            while not done:
                prev_dc_states_map = {dc.id: Observer.get_dc_state(dc, env.sfc_manager, None) 
                                      for dc in env.dcs if dc.is_server}
                
                mask = env._get_valid_actions_mask()
                valid = np.where(mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else 0
                
                _, _, done, _, _ = env.step(action)
                
                active_reqs = Observer.get_active_requests(env.sfc_manager)
                global_stats = Observer.precompute_global_stats(env.sfc_manager, active_reqs)
                
                for dc in env.dcs:
                    if dc.is_server:
                        prev_s = prev_dc_states_map[dc.id]
                        curr_s = Observer.get_dc_state(dc, env.sfc_manager, global_stats)
                        value = Observer.calculate_dc_value(
                            dc, env.sfc_manager, prev_s, global_stats
                        ) 
                        trainer.collect_transition(prev_s, curr_s, value)
            
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"{trainer.size} samples", end="\r", flush=True)
        
        print(f"\n\nCollected {trainer.size} transitions")
        trainer.train_vae()
        trainer.train_value_network()
        
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/vae_model")
        np.savez("models/vae_model_norm.npz", 
                 mean=trainer.agent.value_mean, 
                 std=trainer.agent.value_std)
        
        print("✓ VAE model saved")
        return trainer.agent

    @classmethod
    def evaluate(self, agent, dc_selector, num_episodes=None, prefix=""):
        """Evaluate agent performance"""
        if num_episodes is None:
            num_episodes = config.TEST_EPISODES
        
        env = self.create_env(dc_selector=dc_selector)
        
        print(f"\n{'='*80}\nEvaluating {prefix or 'model'}: {num_episodes} episodes\n{'='*80}")
        
        ars, delays, throughputs = [], [], []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.select_action(state, 0.0, mask)
                state, _, done, _, info = env.step(action)
            
            stats = env.sfc_manager.get_statistics()
            ars.append(stats['acceptance_ratio'])
            delays.append(stats['avg_e2e_delay'])
            
            completed = env.sfc_manager.completed_history
            throughputs.append(sum(r.bandwidth for r in completed))
            
            print(f"  Episode {ep+1}/{num_episodes}: AR={ars[-1]:.1f}%", 
                  end="\r", flush=True)
        
        print(f"\n\nResults:")
        print(f"  Acceptance: {np.mean(ars):.2f}% ± {np.std(ars):.2f}%")
        print(f"  E2E Delay: {np.mean(delays):.2f} ± {np.std(delays):.2f} ms")
        print(f"  Throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f} Mbps")
        
        self._plot_eval(ars, delays, throughputs, f"fig/{prefix}eval.png")
        
        return {
            'acceptance': (np.mean(ars), np.std(ars)),
            'delay': (np.mean(delays), np.std(delays)),
            'throughput': (np.mean(throughputs), np.std(throughputs))
        }

    def _plot_training(self, rewards, acceptances, path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            eps = range(1, len(acceptances) + 1)
            
            ax1.plot(eps, acceptances, alpha=0.3, color='blue')
            window = min(20, len(acceptances) // 10)
            if len(acceptances) >= window:
                ma = np.convolve(acceptances, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(acceptances)+1), ma, 'r-', linewidth=2)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Acceptance Ratio (%)')
            ax1.set_title('Acceptance Ratio')
            ax1.grid(alpha=0.3)
            
            ax2.plot(eps, rewards, alpha=0.3, color='green')
            if len(rewards) >= window:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax2.plot(range(window, len(rewards)+1), ma, 'r-', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Total Reward')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Plot saved: {path}")
        except Exception as e:
            print(f"  Plot error: {e}")
            
    def _plot_eval(self, ars, delays, throughputs, path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].hist(ars, bins=20, color='blue', alpha=0.7, edgecolor='black')
            axes[0].axvline(np.mean(ars), color='red', linestyle='--',
                           label=f'Mean: {np.mean(ars):.2f}%')
            axes[0].set_xlabel('Acceptance Ratio (%)')
            axes[0].set_title('Acceptance Distribution')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].hist(delays, bins=20, color='green', alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(delays), color='red', linestyle='--',
                           label=f'Mean: {np.mean(delays):.2f} ms')
            axes[1].set_xlabel('E2E Delay (ms)')
            axes[1].set_title('Delay Distribution')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            axes[2].hist(throughputs, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[2].axvline(np.mean(throughputs), color='red', linestyle='--',
                           label=f'Mean: {np.mean(throughputs):.2f} Mbps')
            axes[2].set_xlabel('Throughput (Mbps)')
            axes[2].set_title('Throughput Distribution')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Plot saved: {path}")
        except Exception as e:
            print(f"  Plot error: {e}")