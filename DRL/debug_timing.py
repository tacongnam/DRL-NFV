"""
Debug script to identify timing bottleneck
"""
import time
import numpy as np
from env.sfc_environment import SFCEnvironment, TIMING_STATS
from agent import DQNAgent
from config import DRL_CONFIG, VNF_LIST, SFC_TYPES

def debug_timing():
    print("="*70)
    print("TIMING ANALYSIS - Step by Step Performance")
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
    
    print("\nStarting episode with detailed timing...\n")
    episode_start = time.time()
    
    while not done and step_count < 50:  # Limit to 50 steps for faster analysis
        step_start = time.time()
        
        action = agent.select_action(state, training=True)
        next_state, reward, done, _, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
        step_count += 1
        
        step_time = time.time() - step_start
        
        if step_count % 10 == 0:
            print(f"\nStep {step_count}: {step_time:.3f}s")
            print(f"  Pending: {len(env.pending_sfcs)}, Active: {len(env.active_sfcs)}")
            print(f"  Satisfied: {len(env.satisfied_sfcs)}, Dropped: {len(env.dropped_sfcs)}")
            
            # Print timing stats
            if TIMING_STATS['pathfinding_calls'] > 0:
                avg_pathfinding = TIMING_STATS['pathfinding'] / TIMING_STATS['pathfinding_calls']
                print(f"  Pathfinding: {TIMING_STATS['pathfinding']:.3f}s ({TIMING_STATS['pathfinding_calls']} calls, avg {avg_pathfinding*1000:.2f}ms/call)")
            
            print(f"  Update SFCs: {TIMING_STATS['update_sfcs']:.3f}s")
            print(f"  Check Completion: {TIMING_STATS['check_completion']:.3f}s")
            print(f"  Get Observation: {TIMING_STATS['get_observation']:.3f}s")
            print(f"  Step Total: {TIMING_STATS['step_total']:.3f}s")
    
    total_time = time.time() - episode_start
    
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Total time: {total_time:.3f}s")
    print(f"Steps completed: {step_count}")
    print(f"Time per step: {total_time/max(1,step_count)*1000:.1f}ms")
    print(f"\nDetailed breakdown:")
    
    if TIMING_STATS['pathfinding_calls'] > 0:
        avg_pathfinding = TIMING_STATS['pathfinding'] / TIMING_STATS['pathfinding_calls']
        print(f"  Pathfinding: {TIMING_STATS['pathfinding']:.3f}s ({TIMING_STATS['pathfinding_calls']} calls, avg {avg_pathfinding*1000:.2f}ms/call)")
    
    print(f"  Update SFCs: {TIMING_STATS['update_sfcs']:.3f}s")
    print(f"  Check Completion: {TIMING_STATS['check_completion']:.3f}s")
    print(f"  Get Observation: {TIMING_STATS['get_observation']:.3f}s")
    print(f"  Step Total: {TIMING_STATS['step_total']:.3f}s")
    
    print(f"\nFinal state:")
    print(f"  Pending: {len(env.pending_sfcs)}, Active: {len(env.active_sfcs)}")
    print(f"  Satisfied: {len(env.satisfied_sfcs)}, Dropped: {len(env.dropped_sfcs)}")
    print(f"  Reward: {episode_reward:.2f}")
    print("="*70)

if __name__ == "__main__":
    debug_timing()
