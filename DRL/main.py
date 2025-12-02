import numpy as np
import matplotlib.pyplot as plt
from env.sfc_environment import SFCEnvironment
from agent import DQNAgent
from baseline_heuristic import compare_with_baseline
from config import DRL_CONFIG, VNF_LIST, SFC_TYPES
import os

def train_agent(num_dcs=4, save_path='models/dqn_sfc.weights.h5'):
    env = SFCEnvironment(num_dcs=num_dcs)
    
    state1_dim = 2 * len(VNF_LIST) + 2
    state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
    state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
    action_dim = 2 * len(VNF_LIST) + 1
    
    agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
    
    episode_rewards = []
    acceptance_ratios = []
    losses = []
    avg_delays = []
    resource_utils = []
    
    total_episodes = 0
    
    for update in range(DRL_CONFIG['updates']):
        print(f"\nUpdate {update + 1}/{DRL_CONFIG['updates']}")
        
        update_rewards = []
        update_ratios = []
        update_losses = []
        update_delays = []
        update_utils = []
        
        for episode in range(DRL_CONFIG['episodes_per_update']):
            current_num_dcs = np.random.randint(2, 9)
            env = SFCEnvironment(num_dcs=current_num_dcs)
            
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            done = False
            while not done:
                for _ in range(min(DRL_CONFIG['actions_per_step'], 50)):
                    action = agent.select_action(state, training=True)
                    next_state, reward, done, _, info = env.step(action)
                    
                    agent.store_transition(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    state = next_state
                    
                    loss = agent.train()
                    if loss is not None:
                        update_losses.append(loss)
                    
                    if done:
                        break
                
                step_count += 1
                if step_count > 200:
                    break
            
            update_rewards.append(episode_reward)
            update_ratios.append(info['acceptance_ratio'])
            update_delays.append(info['avg_delay'])
            update_utils.append(info['resource_util'])
            
            total_episodes += 1
            if total_episodes % 10 == 0:
                print(f"Ep {total_episodes}: Reward={episode_reward:.2f}, "
                      f"AccRatio={info['acceptance_ratio']:.2%}, "
                      f"Delay={info['avg_delay']:.2f}ms, "
                      f"ResUtil={info['resource_util']:.2%}, "
                      f"Eps={agent.epsilon:.3f}")
        
        agent.decay_epsilon()
        
        episode_rewards.append(np.mean(update_rewards))
        acceptance_ratios.append(np.mean(update_ratios))
        avg_delays.append(np.mean([d for d in update_delays if d > 0]))
        resource_utils.append(np.mean(update_utils))
        if update_losses:
            losses.append(np.mean(update_losses))
        
        if (update + 1) % 50 == 0:
            os.makedirs('models', exist_ok=True)
            agent.save(save_path)
            print(f"Model saved at update {update + 1}")
    
    agent.save(save_path)
    print(f"\nTraining completed. Final model saved to {save_path}")
    
    return agent, episode_rewards, acceptance_ratios, losses, avg_delays, resource_utils

def test_agent(agent, num_dcs_list=[2, 4, 6, 8], num_tests=5):
    results = {}
    sfc_type_results = {}
    
    for num_dcs in num_dcs_list:
        print(f"\nTesting with {num_dcs} DCs...")
        env = SFCEnvironment(num_dcs=num_dcs)
        
        acceptance_ratios = []
        avg_delays = []
        resource_utilization = []
        sfc_type_stats = {sfc_type: {'satisfied': 0, 'total': 0} for sfc_type in SFC_TYPES.keys()}
        
        for test in range(num_tests):
            state, _ = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 200:
                for _ in range(min(DRL_CONFIG['actions_per_step'], 50)):
                    action = agent.select_action(state, training=False)
                    state, _, done, _, info = env.step(action)
                    
                    if done:
                        break
                
                step_count += 1
            
            acceptance_ratios.append(info['acceptance_ratio'])
            avg_delays.append(info['avg_delay'])
            resource_utilization.append(info['resource_util'])
            
            for sfc in env.satisfied_sfcs:
                sfc_type_stats[sfc['type']]['satisfied'] += 1
                sfc_type_stats[sfc['type']]['total'] += 1
            
            for sfc in env.dropped_sfcs:
                sfc_type_stats[sfc['type']]['total'] += 1
        
        results[num_dcs] = {
            'acceptance_ratio': np.mean(acceptance_ratios),
            'avg_delay': np.mean([d for d in avg_delays if d > 0]) if any(d > 0 for d in avg_delays) else 0,
            'resource_util': np.mean(resource_utilization),
            'std_acc': np.std(acceptance_ratios)
        }
        
        sfc_type_results[num_dcs] = {}
        for sfc_type, stats in sfc_type_stats.items():
            if stats['total'] > 0:
                sfc_type_results[num_dcs][sfc_type] = stats['satisfied'] / stats['total']
            else:
                sfc_type_results[num_dcs][sfc_type] = 0
        
        print(f"  AccRatio: {results[num_dcs]['acceptance_ratio']:.2%} ± {results[num_dcs]['std_acc']:.2%}")
        print(f"  Avg Delay: {results[num_dcs]['avg_delay']:.2f} ms")
        print(f"  Resource Util: {results[num_dcs]['resource_util']:.2%}")
        print(f"  SFC Type Acceptance:")
        for sfc_type, ratio in sfc_type_results[num_dcs].items():
            print(f"    {sfc_type}: {ratio:.2%}")
    
    return results, sfc_type_results

def plot_results(episode_rewards, acceptance_ratios, losses, avg_delays, resource_utils, test_results, sfc_type_results):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episode_rewards, linewidth=2)
    ax1.set_xlabel('Update', fontsize=11)
    ax1.set_ylabel('Average Reward', fontsize=11)
    ax1.set_title('Training Rewards Over Updates', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(acceptance_ratios, linewidth=2, color='green')
    ax2.set_xlabel('Update', fontsize=11)
    ax2.set_ylabel('Acceptance Ratio', fontsize=11)
    ax2.set_title('Training Acceptance Ratio', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    if losses:
        ax3.plot(losses, linewidth=2, color='red')
        ax3.set_xlabel('Update', fontsize=11)
        ax3.set_ylabel('Loss', fontsize=11)
        ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    if avg_delays:
        ax4.plot(avg_delays, linewidth=2, color='orange')
        ax4.set_xlabel('Update', fontsize=11)
        ax4.set_ylabel('Avg E2E Delay (ms)', fontsize=11)
        ax4.set_title('Average E2E Delay', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    if resource_utils:
        ax5.plot(resource_utils, linewidth=2, color='purple')
        ax5.set_xlabel('Update', fontsize=11)
        ax5.set_ylabel('Resource Utilization', fontsize=11)
        ax5.set_title('Resource Utilization', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    num_dcs = list(test_results.keys())
    acc_ratios = [test_results[n]['acceptance_ratio'] for n in num_dcs]
    colors_acc = ['#2ecc71' if r > 0.8 else '#f39c12' if r > 0.6 else '#e74c3c' for r in acc_ratios]
    bars = ax6.bar(num_dcs, acc_ratios, color=colors_acc, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Number of DCs', fontsize=11)
    ax6.set_ylabel('Acceptance Ratio', fontsize=11)
    ax6.set_title('Acceptance Ratio vs Network Size', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax7 = fig.add_subplot(gs[2, 0])
    delays = [test_results[n]['avg_delay'] for n in num_dcs]
    ax7.plot(num_dcs, delays, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax7.set_xlabel('Number of DCs', fontsize=11)
    ax7.set_ylabel('Avg E2E Delay (ms)', fontsize=11)
    ax7.set_title('E2E Delay vs Network Size', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    utils = [test_results[n]['resource_util'] for n in num_dcs]
    ax8.plot(num_dcs, utils, marker='s', linewidth=2, markersize=8, color='#9b59b6')
    ax8.set_xlabel('Number of DCs', fontsize=11)
    ax8.set_ylabel('Resource Utilization', fontsize=11)
    ax8.set_title('Resource Utilization vs Network Size', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    ax9 = fig.add_subplot(gs[2, 2])
    if 4 in sfc_type_results:
        sfc_types = list(sfc_type_results[4].keys())
        ratios = [sfc_type_results[4][t] for t in sfc_types]
        colors_sfc = plt.cm.Set3(range(len(sfc_types)))
        bars = ax9.bar(range(len(sfc_types)), ratios, color=colors_sfc, edgecolor='black', linewidth=1.5)
        ax9.set_xticks(range(len(sfc_types)))
        ax9.set_xticklabels(sfc_types, rotation=45, ha='right')
        ax9.set_ylabel('Acceptance Ratio', fontsize=11)
        ax9.set_title('SFC Type Acceptance (4 DCs)', fontsize=12, fontweight='bold')
        ax9.set_ylim([0, 1.1])
        ax9.grid(True, axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0%}', ha='center', va='bottom', fontsize=8)
    
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to training_results.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("DRL-based SFC Provisioning with Reconfigurability")
    print("="*60)
    print("\nTraining Configuration:")
    print(f"  Updates: {DRL_CONFIG['updates']}")
    print(f"  Episodes per Update: {DRL_CONFIG['episodes_per_update']}")
    print(f"  Actions per Step: {DRL_CONFIG['actions_per_step']}")
    print(f"  Batch Size: {DRL_CONFIG['batch_size']}")
    print(f"  Memory Size: {DRL_CONFIG['memory_size']}")
    print("\nStarting training...\n")
    
    agent, rewards, acc_ratios, losses, delays, utils = train_agent(num_dcs=4)
    
    print("\n" + "="*60)
    print("Testing trained agent on different network configurations")
    print("="*60)
    
    test_results, sfc_type_results = test_agent(agent, num_dcs_list=[2, 4, 6, 8])
    
    print("\n" + "="*60)
    print("Baseline Comparison")
    print("="*60)
    
    drl_baseline, baseline_baseline = compare_with_baseline(agent, num_dcs=4, num_tests=5)
    
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print("\nNetwork Reconfigurability Test:")
    for num_dcs, result in test_results.items():
        print(f"\n  {num_dcs} DCs:")
        print(f"    Acceptance Ratio: {result['acceptance_ratio']:.2%}")
        print(f"    Avg E2E Delay: {result['avg_delay']:.2f} ms")
        print(f"    Resource Utilization: {result['resource_util']:.2%}")
    
    plot_results(rewards, acc_ratios, losses, delays, utils, test_results, sfc_type_results)
    
    print("\n" + "="*60)
    print("Training and testing completed successfully!")
    print("="*60)