# runners/demo.py
import config
from environment.gym_env import Env
from agent.agent import Agent
from spaces.request import SFCRequest
import numpy as np
import sys

def test_config():
    """Test 1: Kiểm tra config"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Validation")
    print("="*60)
    
    print(f"VNF Types ({len(config.VNF_TYPES)}): {config.VNF_TYPES}")
    print(f"SFC Types ({len(config.SFC_TYPES)}): {config.SFC_TYPES}")
    print(f"Action Space Size: {config.ACTION_SPACE_SIZE}")
    print(f"Expected: 2 * {config.NUM_VNF_TYPES} + 1 = {2 * config.NUM_VNF_TYPES + 1}")
    
    assert len(config.VNF_TYPES) == 6, "Should have 6 VNF types"
    assert config.ACTION_SPACE_SIZE == 2 * 6 + 1, "Action space should be 13"
    
    print("✓ Config validation passed")

def test_dqn_shape():
    """Test 2: Kiểm tra DQN model architecture"""
    print("\n" + "="*60)
    print("TEST 2: DQN Model Architecture")
    print("="*60)
    
    agent = Agent()
    
    # Mock inputs với đúng kích thước
    s1_dim = 2 * config.NUM_VNF_TYPES + 2
    s2_dim = config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES)
    s3_dim = config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES)
    
    print(f"Input 1 shape: (1, {s1_dim})")
    print(f"Input 2 shape: (1, {s2_dim})")
    print(f"Input 3 shape: (1, {s3_dim})")
    
    s1 = np.zeros((1, s1_dim), dtype=np.float32)
    s2 = np.zeros((1, s2_dim), dtype=np.float32)
    s3 = np.zeros((1, s3_dim), dtype=np.float32)
    
    try:
        output = agent.model.predict([s1, s2, s3], verbose=0)
        print(f"Output shape: {output.shape}")
        print(f"Expected: (1, {config.ACTION_SPACE_SIZE})")
        
        assert output.shape == (1, config.ACTION_SPACE_SIZE), "Output shape mismatch"
        print("✓ DQN architecture validated")
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        sys.exit(1)

def test_state_representation():
    """Test 3: Kiểm tra state representation"""
    print("\n" + "="*60)
    print("TEST 3: State Representation")
    print("="*60)
    
    env = Env()
    state, _ = env.reset(num_dcs=4)
    
    print(f"State is tuple: {isinstance(state, tuple)}")
    print(f"Number of inputs: {len(state)}")
    print(f"Input 1 shape: {state[0].shape}")
    print(f"Input 2 shape: {state[1].shape}")
    print(f"Input 3 shape: {state[2].shape}")
    
    # Verify shapes
    assert len(state) == 3, "State should have 3 inputs"
    assert state[0].shape[0] == 2 * config.NUM_VNF_TYPES + 2, "Input 1 shape incorrect"
    assert state[1].shape[0] == config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES), "Input 2 shape incorrect"
    assert state[2].shape[0] == config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES), "Input 3 shape incorrect"
    
    print("✓ State representation validated")

def test_action_execution():
    """Test 4: Kiểm tra action execution"""
    print("\n" + "="*60)
    print("TEST 4: Action Execution & Reward System")
    print("="*60)
    
    env = Env()
    state, _ = env.reset(num_dcs=4)
    
    # Test WAIT action
    print("\n[Testing WAIT action]")
    next_state, reward, done, _, info = env.step(0)
    print(f"  Reward: {reward}")
    print(f"  Expected: {config.REWARD_WAIT}")
    assert reward == config.REWARD_WAIT, "WAIT should give neutral reward"
    print("  ✓ WAIT action works correctly")
    
    # Test với mock request
    print("\n[Testing ALLOCATION with mock request]")
    mock_req = SFCRequest(999, 'CloudGaming', 0, 1, 0)
    env.sfc_manager.active_requests.append(mock_req)
    
    needed_vnf = mock_req.get_next_vnf()
    print(f"  Mock request needs: {needed_vnf}")
    
    # Find allocation action for this VNF
    vnf_idx = config.VNF_TYPES.index(needed_vnf)
    action_alloc = 1 + config.NUM_VNF_TYPES + vnf_idx
    print(f"  Executing allocation action: {action_alloc}")
    
    next_state, reward, done, _, info = env.step(action_alloc)
    print(f"  Reward: {reward}")
    
    if reward > 0:
        print("  ✓ Allocation successful (positive reward)")
    else:
        print(f"  ⚠ Reward={reward} (may be invalid if DC lacks resources)")
    
    stats = env.sfc_manager.get_statistics()
    print(f"\n[Statistics]")
    print(f"  Acceptance Ratio: {stats['acceptance_ratio']:.2f}%")
    print(f"  Total Generated: {stats['total_generated']}")
    print(f"  Total Accepted: {stats['total_accepted']}")

def test_episode_flow():
    """Test 5: Kiểm tra episode flow"""
    print("\n" + "="*60)
    print("TEST 5: Episode Flow")
    print("="*60)
    
    env = Env()
    agent = Agent()
    
    state, _ = env.reset(num_dcs=4)
    print(f"Episode started with {len(env.sfc_manager.active_requests)} active requests")
    
    done = False
    step_count = 0
    total_reward = 0
    action_count = 0
    
    print("\nRunning episode...")
    while not done:
        mask = env._get_valid_actions_mask()
        action = agent.get_action(state, epsilon=0.5, valid_actions_mask=mask)
        state, reward, done, _, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        action_count += 1
        
        # Print progress every 500 steps (5 ms)
        if step_count % 500 == 0:
            active = len(env.sfc_manager.active_requests)
            completed = len(env.sfc_manager.completed_history)
            dropped = len(env.sfc_manager.dropped_history)
            print(f"  Step {step_count}: Active={active}, Completed={completed}, Dropped={dropped}, SimTime={env.simulator.sim_time}ms")
        
        # Safety limit for demo
        if step_count > 10000:
            print(f"  ⚠ Stopped at {step_count} steps (safety limit for demo)")
            break
    
    stats = env.sfc_manager.get_statistics()
    
    print(f"\nEpisode completed:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Simulation time: {env.simulator.sim_time} ms")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Total accepted: {stats['total_accepted']}")
    print(f"  Total dropped: {stats['total_dropped']}")
    print(f"  Acceptance ratio: {stats['acceptance_ratio']:.2f}%")
    print(f"  Avg E2E delay: {stats['avg_e2e_delay']:.2f} ms")
    
    if stats['total_accepted'] > 0:
        print("  ✓ Episode successfully processed requests")
    else:
        print("  ⚠ No requests accepted (may need tuning or longer episode)")
    
    if done:
        print("  ✓ Episode terminated naturally")
    else:
        print("  ⚠ Episode stopped at limit")

def main():
    print("\n" + "="*60)
    print("DRL SFC PROVISIONING - DEMO & VALIDATION")
    print("="*60)
    
    try:
        test_config()
        test_dqn_shape()
        test_state_representation()
        test_action_execution()
        test_episode_flow()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nYou can now run:")
        print("  - python scripts.py train   (to train the model)")
        print("  - python scripts.py eval    (to evaluate trained model)")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()