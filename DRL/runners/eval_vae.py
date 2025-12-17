import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import config
from envs.vae_env import VAEEnv
from agents.dqn_agent import Agent
from agents.vae_model import VAEModel
from envs.observer import Observer
from runners.experiments import plot_exp1_results, plot_exp2_results

def run_episode_eval(env, agent):
    """Run evaluation episode"""
    state, _ = env.reset()
    done = False
    
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.get_action(state, epsilon=0.0, valid_actions_mask=mask)
        state, _, done, _, _ = env.step(action)
    
    return env.sfc_manager.get_statistics()

def experiment_performance(env, agent, episodes=None):
    """Exp 1: Performance per SFC"""
    if episodes is None:
        episodes = config.TEST_EPISODES
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Performance (GenAI-DRL), {episodes} episodes")
    print(f"{'='*80}")
    
    total_completed = []
    total_dropped = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        ep_start = time.time()
        env.reset(num_dcs=4)
        stats = run_episode_eval(env, agent)
        
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
        
        ep_time = time.time() - ep_start
        print(f"  Ep {ep+1}/{episodes}: AR={stats['acceptance_ratio']:.1f}% | time={ep_time:.1f}s")
    
    exp_time = time.time() - start_time
    
    print(f"\nResults (completed in {exp_time:.1f}s):")
    
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    for sfc_type in sfc_types:
        completed = [r for r in total_completed if r.type == sfc_type]
        dropped = [r for r in total_dropped if r.type == sfc_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.get_total_e2e_delay() for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        
        print(f"  {sfc_type:15s}: AR={ar:6.2f}% | E2E={avg_delay:6.2f}ms")
    
    plot_exp1_results(sfc_types, acc_ratios, e2e_delays, 
                     save_path='fig/genai_result_exp1.png')

def experiment_scalability(env, agent, episodes=None):
    """Exp 2: Scalability"""
    if episodes is None:
        episodes = config.TEST_EPISODES
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: Scalability (GenAI-DRL), {episodes} episodes/config")
    print(f"{'='*80}")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    exp_start = time.time()
    
    for n_dc in dc_counts:
        print(f"\n[Config: {n_dc} DCs]")
        config_start = time.time()
        
        current_completed = []
        cpu_usages = []
        
        for ep in range(episodes):
            ep_start = time.time()
            env.reset(num_dcs=n_dc)
            
            state, _ = env._get_obs(), {}
            done = False
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=0.0, valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                total_cap = n_dc * config.DC_CPU_CYCLES
                used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in env.dcs)
                usage_pct = (used_cap / total_cap * 100) if total_cap > 0 else 0
                cpu_usages.append(usage_pct)
            
            current_completed.extend(env.sfc_manager.completed_history)
            
            ep_time = time.time() - ep_start
            print(f"  Ep {ep+1}/{episodes}: time={ep_time:.1f}s")
        
        avg_delay = np.mean([r.get_total_e2e_delay() for r in current_completed]) \
                    if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        
        config_time = time.time() - config_start
        print(f"  → E2E={avg_delay:.2f}ms | CPU={avg_cpu:.2f}% | time={config_time:.1f}s")
    
    exp_time = time.time() - exp_start
    
    plot_exp2_results(dc_counts, exp2_delays, exp2_resources,
                     save_path='fig/genai_result_exp2.png')
    
    print(f"\nExp 2 completed in {exp_time/60:.1f} min")

def main():
    print("\n" + "="*80)
    print("EVALUATING GenAI-DRL MODEL")
    print("="*80)
    
    start_time = time.time()
    
    # Load GenAI
    state_dim = Observer.get_state_dim()
    genai_model = VAEModel(state_dim, latent_dim=config.GENAI_LATENT_DIM)
    
    genai_path = 'models/genai_model'
    if os.path.exists(f'{genai_path}_encoder.weights.h5'):
        print(f"Loading GenAI: {genai_path}")
        genai_model.load_weights(genai_path)
        print("✓ GenAI loaded")
    else:
        print(f"\n✗ GenAI not found!")
        return
    
    env = VAEEnv(genai_model=genai_model, data_collection_mode=False)
    agent = Agent()
    
    # Load DRL
    weights_path = f'models/genai_{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"Loading DRL: {weights_path}")
        dummy_state, _ = env.reset()
        agent.get_action(dummy_state, 0.0)
        agent.model.load_weights(weights_path)
        print("✓ DRL loaded")
    else:
        print(f"\n✗ DRL weights not found!")
        return
    
    print(f"\nTest Config: {config.TEST_EPISODES} episodes")
    
    # Experiments
    experiment_performance(env, agent)
    experiment_scalability(env, agent)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETED in {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("="*80)
    print("Check fig/ for plots")

if __name__ == "__main__":
    main()