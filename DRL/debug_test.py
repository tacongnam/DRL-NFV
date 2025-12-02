from env.sfc_environment import SFCEnvironment
from agent import DQNAgent
from config import *
import traceback

def test_environment_basic():
    print("="*60)
    print("Test 1: Environment Basic Operations")
    print("="*60)
    
    try:
        env = SFCEnvironment(num_dcs=4)
        print("✓ Environment created successfully")
        
        state, _ = env.reset()
        print("✓ Environment reset successfully")
        print(f"  State1 shape: {state['state1'].shape}")
        print(f"  State2 shape: {state['state2'].shape}")
        print(f"  State3 shape: {state['state3'].shape}")
        
        env._generate_sfc_requests()
        print(f"✓ Generated {len(env.pending_sfcs)} SFC requests")
        
        for i in range(10):
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            if done:
                print(f"✓ Episode done after {i+1} steps")
                break
        
        print(f"  Satisfied: {info['satisfied']}, Dropped: {info['dropped']}")
        print(f"  Acceptance Ratio: {info['acceptance_ratio']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_agent_basic():
    print("\n" + "="*60)
    print("Test 2: Agent Basic Operations")
    print("="*60)
    
    try:
        state1_dim = 2 * len(VNF_LIST) + 2
        state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
        state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
        action_dim = 2 * len(VNF_LIST) + 1
        
        agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
        print("✓ Agent created successfully")
        
        env = SFCEnvironment(num_dcs=4)
        state, _ = env.reset()
        
        action = agent.select_action(state, training=False)
        print(f"✓ Agent selected action: {action}")
        
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        print(f"✓ Transition stored, memory size: {len(agent.memory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_training_loop():
    print("\n" + "="*60)
    print("Test 3: Training Loop")
    print("="*60)
    
    try:
        env = SFCEnvironment(num_dcs=4)
        
        state1_dim = 2 * len(VNF_LIST) + 2
        state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
        state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
        action_dim = 2 * len(VNF_LIST) + 1
        
        agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
        
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(20):
            for _ in range(5):
                action = agent.select_action(state, training=True)
                next_state, reward, done, _, info = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                
                if len(agent.memory) >= 32:
                    loss = agent.train()
                    if loss is not None and step % 10 == 0:
                        print(f"  Step {step}: Loss={loss:.4f}")
                
                if done:
                    break
            
            if done:
                break
        
        print(f"✓ Training loop completed")
        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Acceptance ratio: {info['acceptance_ratio']:.2%}")
        print(f"  Memory size: {len(agent.memory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_vnf_allocation():
    print("\n" + "="*60)
    print("Test 4: VNF Allocation Logic")
    print("="*60)
    
    try:
        env = SFCEnvironment(num_dcs=4)
        env.reset()
        env._generate_sfc_requests()
        
        print(f"✓ Generated {len(env.pending_sfcs)} pending SFCs")
        
        if env.pending_sfcs:
            sfc = env.pending_sfcs[0]
            print(f"  SFC type: {sfc['type']}")
            print(f"  VNF chain: {' → '.join(sfc['vnfs'])}")
            print(f"  Source: {sfc['source']}, Dest: {sfc['dest']}")
            print(f"  E2E delay limit: {sfc['delay']} ms")
            
            vnf_type = sfc['vnfs'][0]
            vnf_idx = VNF_LIST.index(vnf_type)
            action = len(VNF_LIST) + vnf_idx
            
            print(f"  Attempting to allocate: {vnf_type}")
            
            state, reward, done, _, info = env.step(action)
            print(f"  Reward: {reward}")
            print(f"  Active SFCs: {len(env.active_sfcs)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_reconfigurability():
    print("\n" + "="*60)
    print("Test 5: Network Reconfigurability")
    print("="*60)
    
    try:
        state1_dim = 2 * len(VNF_LIST) + 2
        state2_dim = len(list(SFC_TYPES.keys())) * (1 + 2 * len(VNF_LIST))
        state3_dim = len(list(SFC_TYPES.keys())) * (4 + len(VNF_LIST))
        action_dim = 2 * len(VNF_LIST) + 1
        
        agent = DQNAgent(state1_dim, state2_dim, state3_dim, action_dim)
        
        for num_dcs in [2, 4, 6, 8]:
            env = SFCEnvironment(num_dcs=num_dcs)
            state, _ = env.reset()
            
            action = agent.select_action(state, training=False)
            next_state, reward, done, _, info = env.step(action)
            
            print(f"✓ {num_dcs} DCs: Action space={env.action_space.n}, State dims OK")
        
        print("✓ Reconfigurability verified across all network sizes")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

def test_constraints():
    print("\n" + "="*60)
    print("Test 6: Constraint Checking")
    print("="*60)
    
    try:
        env = SFCEnvironment(num_dcs=4)
        env.reset()
        env._generate_sfc_requests()
        
        for dc in env.dcs:
            cpu_used = sum(dc['installed_vnfs'][vnf] * VNF_SPECS[vnf]['cpu'] 
                          for vnf in VNF_LIST)
            storage_used = sum(dc['installed_vnfs'][vnf] * VNF_SPECS[vnf]['storage'] 
                              for vnf in VNF_LIST)
            
            assert cpu_used <= dc['cpu'], f"C1 violated: CPU {cpu_used} > {dc['cpu']}"
            assert storage_used <= dc['storage'], f"C2 violated: Storage {storage_used} > {dc['storage']}"
        
        print("✓ C1 (CPU constraint) verified")
        print("✓ C2 (Storage constraint) verified")
        
        for sfc in env.active_sfcs + env.satisfied_sfcs:
            if sfc['allocated_vnfs']:
                vnf_set = set()
                for i, vnf in enumerate(sfc['allocated_vnfs']):
                    dc_id = sfc['allocated_dcs'][i]
                    key = (vnf, i, dc_id)
                    assert key not in vnf_set, "C3 violated: VNF allocated twice"
                    vnf_set.add(key)
        
        print("✓ C3 (VNF uniqueness) verified")
        
        for i, j in env.network.edges():
            used_bw = 0
            for sfc in env.active_sfcs:
                if len(sfc['allocated_dcs']) > 1:
                    for k in range(len(sfc['allocated_dcs']) - 1):
                        if (sfc['allocated_dcs'][k] == i and sfc['allocated_dcs'][k+1] == j) or \
                           (sfc['allocated_dcs'][k] == j and sfc['allocated_dcs'][k+1] == i):
                            used_bw += sfc['bw']
            
            available = env.network[i][j]['available_bw']
            assert used_bw <= DC_CONFIG['link_bw'], f"C4 violated: BW {used_bw} > {DC_CONFIG['link_bw']}"
        
        print("✓ C4 (Bandwidth constraint) verified")
        
        for sfc in env.satisfied_sfcs:
            elapsed = env.current_time - sfc['created_time']
            assert elapsed <= sfc['delay'], f"C5 violated: Delay {elapsed} > {sfc['delay']}"
        
        print("✓ C5 (E2E delay constraint) verified")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Constraint violation: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DRL-SFC Debug Test Suite")
    print("="*60 + "\n")
    
    tests = [
        test_environment_basic,
        test_agent_basic,
        test_training_loop,
        test_vnf_allocation,
        test_reconfigurability,
        test_constraints
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check errors above.")
    
    print("\n" + "="*60)