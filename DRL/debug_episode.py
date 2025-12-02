"""
Debug script to identify what's slowing down episode execution
"""
import time
import numpy as np
from env.sfc_environment import SFCEnvironment
from agent import DQNAgent
from config import DRL_CONFIG, VNF_LIST, SFC_TYPES

def debug_episode():
    print("="*70)
    print("EPISODE DEBUG - Identifying Performance Bottleneck")
    print("="*70)
    
    env = SFCEnvironment(num_dcs=4)
    
    state1_dim = 2 * len(VNF_LIST) + 2
    state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
    state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
    action_dim = 2 * len(VNF_LIST) + 1
    
    agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
    
    state, _ = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    step_times = []
    step_details = []
    
    print("\nStarting episode...")
    episode_start = time.time()
    
    while not done and step_count < 200:
        step_start = time.time()
        
        for inner_step in range(min(DRL_CONFIG['actions_per_step'], 50)):
            inner_start = time.time()
            
            # 1. Action selection
            action_start = time.time()
            action = agent.select_action(state, training=True)
            action_time = time.time() - action_start
            
            # 2. Environment step
            env_start = time.time()
            next_state, reward, done, _, info = env.step(action)
            env_time = time.time() - env_start
            
            # 3. Store transition
            store_start = time.time()
            agent.store_transition(state, action, reward, next_state, done)
            store_time = time.time() - store_start
            
            # 4. Training (OPTIMIZED: Train every train_freq steps)
            train_start = time.time()
            if (step_count * min(DRL_CONFIG['actions_per_step'], 50) + inner_step) % DRL_CONFIG.get('train_freq', 5) == 0:
                if len(agent.memory) >= DRL_CONFIG['batch_size']:
                    loss = agent.train()
            train_time = time.time() - train_start
            
            episode_reward += reward
            state = next_state
            
            inner_total = time.time() - inner_start
            
            if step_count == 0 and inner_step < 3:  # Print first few steps details
                print(f"\n  Inner Step {inner_step}: {inner_total*1000:.1f}ms")
                print(f"    - Action selection: {action_time*1000:.2f}ms")
                print(f"    - Env.step():      {env_time*1000:.2f}ms")
                print(f"    - Store transition: {store_time*1000:.2f}ms")
                print(f"    - Training:        {train_time*1000:.2f}ms")
            """
            elif inner_step > 0 and train_time > 0.1:
                print(f"\n  ⚠ SLOW Inner Step {inner_step}: Training took {train_time*1000:.1f}ms (Memory size: {len(agent.memory)})")
            """
            
            if done:
                break
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        step_count += 1
        
        if step_count % 10 == 0:
            avg_step_time = np.mean(step_times[-10:])
            print(f"\nStep {step_count}: {step_time:.3f}s (avg last 10: {avg_step_time:.3f}s)")
            print(f"  Pending: {len(env.pending_sfcs)}, Active: {len(env.active_sfcs)}")
            print(f"  Satisfied: {len(env.satisfied_sfcs)}, Dropped: {len(env.dropped_sfcs)}")
            
            if step_count > 50 and avg_step_time > 10:
                print("\n  ⚠ WARNING: Step time > 10s - likely infinite loop or pathfinding issue!")
                break
    
    total_time = time.time() - episode_start
    
    print("\n" + "="*70)
    print("EPISODE SUMMARY")
    print("="*70)
    print(f"Total time: {total_time:.3f}s")
    print(f"Steps completed: {step_count}")
    print(f"Time per step: {total_time/max(1,step_count)*1000:.1f}ms")
    print(f"Reward: {episode_reward:.2f}")
    print(f"Acceptance ratio: {info['acceptance_ratio']:.1%}")
    print(f"Final state - Pending: {len(env.pending_sfcs)}, Active: {len(env.active_sfcs)}")
    print(f"Final state - Satisfied: {len(env.satisfied_sfcs)}, Dropped: {len(env.dropped_sfcs)}")
    
    print("\n" + "="*70)
    if total_time < 1:
        print("✓ GOOD - Episode completed in <1 second")
    elif total_time < 5:
        print("✓ ACCEPTABLE - Episode completed in <5 seconds")
    elif total_time < 10:
        print("⚠ SLOW - Episode took >5 seconds")
    else:
        print("✗ VERY SLOW - Episode took >10 seconds")
        print("\nLikely issues:")
        print("  1. Infinite loop in environment logic")
        print("  2. Pathfinding (get_shortest_path_with_bw) still slow")
        print("  3. SFC never completes (stuck in pending/active)")
    
    print("="*70)

if __name__ == "__main__":
    debug_episode()
