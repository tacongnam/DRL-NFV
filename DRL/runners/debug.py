# runners/debug.py
"""
Debug helper để trace chi tiết quá trình xử lý SFC
"""

import config
from environment.gym_env import Env
from agent.agent import Agent

def debug_single_request():
    """Debug chi tiết một request từ đầu đến cuối"""
    print("\n" + "="*80)
    print("DEBUG: Tracking Single Request Processing")
    print("="*80)
    
    env = Env()
    agent = Agent()
    
    # Reset với cấu hình đơn giản
    state, _ = env.reset(num_dcs=2)
    
    # Tạo một request đơn giản
    from spaces.request import SFCRequest
    test_req = SFCRequest(999, 'CloudGaming', 0, 1, 0)
    env.sfc_manager.active_requests.clear()
    env.sfc_manager.active_requests.append(test_req)
    
    print(f"\nTest Request Created:")
    print(f"  ID: {test_req.id}")
    print(f"  Type: {test_req.type}")
    print(f"  Chain: {test_req.chain}")
    print(f"  Source: {test_req.source} → Destination: {test_req.destination}")
    print(f"  Max Delay: {test_req.max_delay} ms")
    print(f"  Chain Length: {len(test_req.chain)} VNFs")
    
    print(f"\nDC Resources:")
    for dc in env.dcs:
        print(f"  DC {dc.id}: CPU={dc.cpu}, Storage={dc.storage}")
    
    print(f"\nStarting Processing...")
    print("-" * 80)
    
    step = 0
    done = False
    
    while step < 1000:
        # Get current state
        vnf_needed = test_req.get_next_vnf()
        
        if vnf_needed and step % 100 == 0:
            print(f"\n[Step {step}] Time: {env.simulator.sim_time}ms")
            print(f"  Request status: VNF {test_req.current_vnf_index}/{len(test_req.chain)}")
            print(f"  Next VNF needed: {vnf_needed}")
            print(f"  Elapsed time: {test_req.elapsed_time:.2f} ms")
            print(f"  Remaining time: {test_req.get_remaining_time():.2f} ms")
            print(f"  Total E2E delay so far: {test_req.get_total_e2e_delay():.2f} ms")
            
            # Check VNF instances in DCs
            for dc in env.dcs:
                idle_count = dc.count_idle_vnf_type(vnf_needed) if vnf_needed else 0
                total_count = dc.count_vnf_type(vnf_needed) if vnf_needed else 0
                if total_count > 0:
                    print(f"  DC {dc.id}: {vnf_needed} instances: {idle_count}/{total_count} idle")
        
        # Select action
        mask = env._get_valid_actions_mask()
        action = agent.get_action(state, epsilon=0.1, valid_actions_mask=mask)
        
        # Execute
        state, reward, done, _, info = env.step(action)
        step += 1
        
        # Check completion
        if test_req.is_completed:
            print(f"\n{'='*80}")
            print(f"✓ REQUEST COMPLETED at step {step}!")
            print(f"{'='*80}")
            print(f"  Total time: {test_req.elapsed_time:.2f} ms")
            print(f"  E2E delay: {test_req.get_total_e2e_delay():.2f} ms")
            print(f"  Propagation delay: {test_req.total_propagation_delay:.2f} ms")
            print(f"  Processing delay: {test_req.total_processing_delay:.2f} ms")
            print(f"  Placed VNFs:")
            for vnf_name, dc_id, prop_d, proc_d in test_req.placed_vnfs:
                print(f"    - {vnf_name} at DC{dc_id}: prop={prop_d:.2f}ms, proc={proc_d:.2f}ms")
            break
        
        if test_req.is_dropped:
            print(f"\n{'='*80}")
            print(f"✗ REQUEST DROPPED at step {step}")
            print(f"{'='*80}")
            print(f"  Elapsed: {test_req.elapsed_time:.2f} ms > Max: {test_req.max_delay} ms")
            print(f"  Progress: {test_req.current_vnf_index}/{len(test_req.chain)} VNFs placed")
            break
    
    if step >= 1000:
        print(f"\n⚠ Stopped at step limit (1000)")
        print(f"  Request status: {test_req.current_vnf_index}/{len(test_req.chain)} VNFs placed")
    
    stats = env.sfc_manager.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Acceptance Ratio: {stats['acceptance_ratio']:.2f}%")
    print(f"  Total Generated: {stats['total_generated']}")
    print(f"  Total Accepted: {stats['total_accepted']}")
    print(f"  Total Dropped: {stats['total_dropped']}")

def debug_vnf_processing():
    """Debug VNF processing và tick mechanism"""
    print("\n" + "="*80)
    print("DEBUG: VNF Processing Mechanism")
    print("="*80)
    
    from spaces.vnf import VNFInstance
    from spaces.dc import DataCenter
    
    # Create test VNF
    vnf = VNFInstance('NAT', 0)
    print(f"\nCreated VNF: {vnf.vnf_type}")
    print(f"Initial state: idle={vnf.is_idle()}, remaining_time={vnf.remaining_proc_time}")
    
    # Assign with processing time
    proc_time = config.VNF_SPECS['NAT']['proc_time']
    print(f"\nAssigning to request 123 with proc_time={proc_time}ms")
    vnf.assign(123, proc_time)
    
    print(f"After assign: idle={vnf.is_idle()}, remaining_time={vnf.remaining_proc_time}")
    
    # Simulate ticks
    print(f"\nSimulating time steps:")
    for i in range(5):
        vnf.tick()
        print(f"  Tick {i+1}: remaining={vnf.remaining_proc_time:.2f}ms, idle={vnf.is_idle()}")
        if vnf.is_idle():
            print(f"  → VNF became idle after {i+1} ticks")
            break

def main():
    """Run all debug tests"""
    print("\n" + "="*80)
    print("DRL SFC PROVISIONING - DEBUG MODE")
    print("="*80)
    
    # Test 1: VNF processing
    debug_vnf_processing()
    
    # Test 2: Single request tracking
    debug_single_request()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()