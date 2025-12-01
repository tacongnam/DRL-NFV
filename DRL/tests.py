import numpy as np
import tensorflow as tf
from env.sfc_environment import SFCEnvironment
from env.dqn_model import DQNModel
from config import *
from utils import generate_sfc_requests
import matplotlib.pyplot as plt

class BaselineHeuristic:
    def __init__(self, env):
        self.env = env
    
    def select_action(self):
        pending_reqs = [r for r in self.env.sfc_requests if r['status'] == 'pending']
        
        if not pending_reqs:
            return 2 * len(VNF_TYPES)
        
        pending_reqs.sort(key=lambda r: r['delay_tolerance'] - (self.env.current_time - r['creation_time']))
        
        target_req = pending_reqs[0]
        next_vnf_idx = len(target_req['allocated_vnfs'])
        
        if next_vnf_idx >= len(target_req['chain']):
            return 2 * len(VNF_TYPES)
        
        vnf_type = target_req['chain'][next_vnf_idx]
        action = VNF_TYPES.index(vnf_type)
        
        return action

def compare_with_baseline(num_episodes=10):
    print("="*60)
    print("COMPARING DRL MODEL VS BASELINE HEURISTIC")
    print("="*60)
    
    drl_model = DQNModel(len(VNF_TYPES), len(SFC_TYPES))
    try:
        drl_model.load('checkpoints/final_model.weights.h5')
        print("Loaded trained DRL model\n")
    except:
        print("No trained model found. Using untrained model.\n")
    
    drl_results = {'acceptance': [], 'delay': [], 'cpu_usage': [], 'storage_usage': []}
    baseline_results = {'acceptance': [], 'delay': [], 'cpu_usage': [], 'storage_usage': []}
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        
        seed = ep * 100
        
        env_drl = SFCEnvironment(num_dcs=4)
        env_drl.reset(seed=seed)
        for i in range(3):
            requests = generate_sfc_requests()
            env_drl.add_requests(requests)
        
        state = env_drl._get_state()
        step = 0
        while step < TRAINING_CONFIG['max_actions_per_step']:
            q_values = drl_model.predict(state)
            action = np.argmax(q_values)
            next_state, _, done, _, _ = env_drl.step(action)
            state = next_state
            step += 1
            if done:
                break
        
        drl_acc = env_drl.get_acceptance_ratio()
        drl_cpu = sum(dc['cpu_used'] for dc in env_drl.dcs) / sum(dc['cpu'] for dc in env_drl.dcs)
        drl_storage = sum(dc['storage_used'] for dc in env_drl.dcs) / sum(dc['storage'] for dc in env_drl.dcs)
        
        delays = []
        for req in env_drl.sfc_requests:
            if req['status'] == 'accepted':
                delays.append(env_drl._calculate_total_delay(req))
        drl_delay = np.mean(delays) if delays else 0
        
        drl_results['acceptance'].append(drl_acc)
        drl_results['delay'].append(drl_delay)
        drl_results['cpu_usage'].append(drl_cpu)
        drl_results['storage_usage'].append(drl_storage)
        
        env_baseline = SFCEnvironment(num_dcs=4)
        env_baseline.reset(seed=seed)
        for i in range(3):
            requests = generate_sfc_requests()
            env_baseline.add_requests(requests)
        
        baseline = BaselineHeuristic(env_baseline)
        step = 0
        while step < TRAINING_CONFIG['max_actions_per_step']:
            action = baseline.select_action()
            _, _, done, _, _ = env_baseline.step(action)
            step += 1
            if done:
                break
        
        baseline_acc = env_baseline.get_acceptance_ratio()
        baseline_cpu = sum(dc['cpu_used'] for dc in env_baseline.dcs) / sum(dc['cpu'] for dc in env_baseline.dcs)
        baseline_storage = sum(dc['storage_used'] for dc in env_baseline.dcs) / sum(dc['storage'] for dc in env_baseline.dcs)
        
        delays = []
        for req in env_baseline.sfc_requests:
            if req['status'] == 'accepted':
                delays.append(env_baseline._calculate_total_delay(req))
        baseline_delay = np.mean(delays) if delays else 0
        
        baseline_results['acceptance'].append(baseline_acc)
        baseline_results['delay'].append(baseline_delay)
        baseline_results['cpu_usage'].append(baseline_cpu)
        baseline_results['storage_usage'].append(baseline_storage)
        
        print(f"DRL - AccRatio: {drl_acc:.3f}, Delay: {drl_delay:.2f}ms")
        print(f"Baseline - AccRatio: {baseline_acc:.3f}, Delay: {baseline_delay:.2f}ms")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    drl_avg_acc = np.mean(drl_results['acceptance'])
    baseline_avg_acc = np.mean(baseline_results['acceptance'])
    improvement_acc = ((drl_avg_acc - baseline_avg_acc) / baseline_avg_acc) * 100
    
    drl_avg_delay = np.mean(drl_results['delay'])
    baseline_avg_delay = np.mean(baseline_results['delay'])
    improvement_delay = ((baseline_avg_delay - drl_avg_delay) / baseline_avg_delay) * 100
    
    drl_avg_cpu = np.mean(drl_results['cpu_usage'])
    baseline_avg_cpu = np.mean(baseline_results['cpu_usage'])
    improvement_cpu = ((baseline_avg_cpu - drl_avg_cpu) / baseline_avg_cpu) * 100
    
    drl_avg_storage = np.mean(drl_results['storage_usage'])
    baseline_avg_storage = np.mean(baseline_results['storage_usage'])
    improvement_storage = ((baseline_avg_storage - drl_avg_storage) / baseline_avg_storage) * 100
    
    print(f"\nAcceptance Ratio:")
    print(f"  DRL:      {drl_avg_acc:.3f}")
    print(f"  Baseline: {baseline_avg_acc:.3f}")
    print(f"  Improvement: {improvement_acc:+.2f}% (Target: +20.3%)")
    
    print(f"\nE2E Delay (ms):")
    print(f"  DRL:      {drl_avg_delay:.2f}")
    print(f"  Baseline: {baseline_avg_delay:.2f}")
    print(f"  Improvement: {improvement_delay:+.2f}% (Target: +42.65%)")
    
    print(f"\nCPU Usage:")
    print(f"  DRL:      {drl_avg_cpu:.3f}")
    print(f"  Baseline: {baseline_avg_cpu:.3f}")
    print(f"  Improvement: {improvement_cpu:+.2f}%")
    
    print(f"\nStorage Usage:")
    print(f"  DRL:      {drl_avg_storage:.3f}")
    print(f"  Baseline: {baseline_avg_storage:.3f}")
    print(f"  Improvement: {improvement_storage:+.2f}% (Target: +50%)")
    
    plot_comparison(drl_results, baseline_results)

def plot_comparison(drl_results, baseline_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].bar(['DRL', 'Baseline'], 
                   [np.mean(drl_results['acceptance']), np.mean(baseline_results['acceptance'])],
                   color=['green', 'orange'])
    axes[0, 0].set_title('Average Acceptance Ratio')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(['DRL', 'Baseline'], 
                   [np.mean(drl_results['delay']), np.mean(baseline_results['delay'])],
                   color=['green', 'orange'])
    axes[0, 1].set_title('Average E2E Delay')
    axes[0, 1].set_ylabel('Delay (ms)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].bar(['DRL', 'Baseline'], 
                   [np.mean(drl_results['cpu_usage']), np.mean(baseline_results['cpu_usage'])],
                   color=['green', 'orange'])
    axes[1, 0].set_title('Average CPU Usage')
    axes[1, 0].set_ylabel('Usage Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(['DRL', 'Baseline'], 
                   [np.mean(drl_results['storage_usage']), np.mean(baseline_results['storage_usage'])],
                   color=['green', 'orange'])
    axes[1, 1].set_title('Average Storage Usage')
    axes[1, 1].set_ylabel('Usage Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    print("\nComparison plot saved as 'comparison_results.png'")
    plt.close()

def test_reconfigurability():
    print("\n" + "="*60)
    print("TESTING RECONFIGURABILITY ACROSS DIFFERENT NCs")
    print("="*60)
    
    model = DQNModel(len(VNF_TYPES), len(SFC_TYPES))
    try:
        model.load('checkpoints/final_model.weights.h5')
        print("Loaded trained model\n")
    except:
        print("No trained model found.\n")
        return
    
    dc_configs = [2, 4, 6, 8]
    results = []
    
    for num_dcs in dc_configs:
        print(f"\nTesting with {num_dcs} DCs...")
        env = SFCEnvironment(num_dcs=num_dcs)
        
        acceptance_ratios = []
        for ep in range(5):
            state, _ = env.reset()
            
            for i in range(3):
                requests = generate_sfc_requests()
                env.add_requests(requests)
            
            step = 0
            while step < TRAINING_CONFIG['max_actions_per_step']:
                q_values = model.predict(state)
                action = np.argmax(q_values)
                next_state, _, done, _, _ = env.step(action)
                state = next_state
                step += 1
                if done:
                    break
            
            acceptance_ratios.append(env.get_acceptance_ratio())
        
        avg_acc = np.mean(acceptance_ratios)
        results.append(avg_acc)
        print(f"  Average Acceptance Ratio: {avg_acc:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(dc_configs, results, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Data Centers')
    plt.ylabel('Acceptance Ratio')
    plt.title('Model Performance Across Different Network Configurations')
    plt.grid(True, alpha=0.3)
    plt.savefig('reconfigurability_test.png', dpi=150)
    print("\nReconfigurability plot saved as 'reconfigurability_test.png'")
    plt.close()

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    
    compare_with_baseline(num_episodes=20)
    test_reconfigurability()