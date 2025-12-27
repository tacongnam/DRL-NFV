"""
Test script to verify DRL-NFV environment with loaded data.
"""
import sys
import numpy as np
from data_info.read_data import Read_data
from DRL.envs.drl_env import DRLEnv
from DRL import config

def test_data_loading():
    """Test loading data from JSON file"""
    print("=" * 60)
    print("Test 1: Data Loading")
    print("=" * 60)
    
    data_path = "data_1_9/cogent_centers_easy_s1.json"
    print(f"Loading data from: {data_path}")
    
    try:
        reader = Read_data(data_path)
        
        # Load network components
        dc_list = reader.get_V()
        topology = reader.get_E(len(dc_list))
        vnf_specs = reader.get_F()
        requests_data = reader.get_R()
        
        print(f"\n✓ Successfully loaded data:")
        print(f"  - Data Centers: {len(dc_list)}")
        print(f"    - Servers: {sum(1 for dc in dc_list if dc.is_server)}")
        print(f"    - Non-servers: {sum(1 for dc in dc_list if not dc.is_server)}")
        print(f"  - VNF Types: {len(vnf_specs)}")
        print(f"  - Requests: {len(requests_data)}")
        
        # Show sample DC info
        print(f"\n  Sample DC info:")
        for i, dc in enumerate(dc_list[:3]):
            if dc.is_server:
                print(f"    DC {dc.id}: Server - CPU={dc.cpu}, RAM={dc.ram}, "
                      f"Storage={dc.storage}, Delay={dc.delay}")
            else:
                print(f"    DC {dc.id}: Non-server node (cannot host VNFs)")
        
        # Show sample VNF info
        print(f"\n  Sample VNF info:")
        for vnf_type, specs in list(vnf_specs.items())[:3]:
            print(f"    VNF Type {vnf_type}: CPU={specs['cpu']}, RAM={specs['ram']}, "
                  f"Storage={specs['storage']}")
            print(f"      Startup times on servers: {list(specs['startup_time'].items())[:2]}")
        
        # Show sample request info
        print(f"\n  Sample Request info:")
        for req in requests_data[:3]:
            print(f"    Request {req['id']}: Arrival={req['arrival_time']}, "
                  f"Chain={req['vnf_chain']}, BW={req['bandwidth']}, MaxDelay={req['max_delay']}")
        
        # Get network statistics
        print(f"\n  Network Statistics:")
        info = reader.get_info_network()
        print(f"    - Total DCs: {info['num_dcs']}")
        print(f"    - Server nodes: {info['num_servers']}")
        print(f"    - VNF types: {info['num_vnf_types']}")
        print(f"    - Total requests: {info['num_requests']}")
        
        if info['server_stats']:
            print(f"    - Server resources:")
            print(f"      Max CPU: {info['server_stats']['max_cpu']}")
            print(f"      Max RAM: {info['server_stats']['max_ram']}")
            print(f"      Total CPU: {info['server_stats']['total_cpu']}")
        
        if info['topology_stats']:
            print(f"    - Topology:")
            print(f"      Links: {info['topology_stats']['num_links']}")
            print(f"      Max BW: {info['topology_stats']['max_bandwidth']}")
            print(f"      Avg Delay: {info['topology_stats']['avg_delay']:.2f} ms")
        
        return dc_list, topology, vnf_specs, requests_data
        
    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_environment_initialization(dc_list, topology, requests_data):
    """Test environment initialization with loaded data"""
    print("\n" + "=" * 60)
    print("Test 2: Environment Initialization")
    print("=" * 60)
    
    try:
        # Create environment with loaded data
        env = DRLEnv(dcs=dc_list, topology=topology, requests_data=requests_data)
        print("✓ Environment created successfully")
        
        # Reset environment
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        
        print(f"\n  Initial state:")
        print(f"    - Active requests: {len(env.sfc_manager.active_requests)}")
        print(f"    - Total loaded requests: {len(env.sfc_manager.all_requests)}")
        print(f"    - Number of DCs: {len(env.dcs)}")
        print(f"    - Current simulation time: {env.simulator.sim_time} ms")
        
        return env
        
    except Exception as e:
        print(f"\n✗ Failed to initialize environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_environment_step(env):
    """Test running environment for a few steps"""
    print("\n" + "=" * 60)
    print("Test 3: Environment Step Execution")
    print("=" * 60)
    
    try:
        max_steps = 50
        print(f"Running {max_steps} steps...")
        
        for step in range(max_steps):
            # Get valid action (simple random policy)
            action = env.action_space.sample()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            if step % 10 == 0:
                print(f"\n  Step {step}:")
                print(f"    - Action: {action}")
                print(f"    - Reward: {reward:.2f}")
                print(f"    - Active requests: {len(env.sfc_manager.active_requests)}")
                print(f"    - Completed: {len(env.sfc_manager.completed_history)}")
                print(f"    - Dropped: {len(env.sfc_manager.dropped_history)}")
                print(f"    - Sim time: {env.simulator.sim_time} ms")
            
            if done:
                print(f"\n  Episode finished at step {step}")
                break
        
        # Final statistics
        stats = env.sfc_manager.get_statistics()
        print(f"\n✓ Environment step test completed")
        print(f"\n  Final Statistics:")
        print(f"    - Total requests: {stats['total_generated']}")
        print(f"    - Accepted: {stats['total_accepted']}")
        print(f"    - Dropped: {stats['total_dropped']}")
        print(f"    - Acceptance ratio: {stats['acceptance_ratio']:.2f}%")
        print(f"    - Drop ratio: {stats['drop_ratio']:.2f}%")
        if stats['avg_e2e_delay'] > 0:
            print(f"    - Avg E2E delay: {stats['avg_e2e_delay']:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed during environment step: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dc_capabilities():
    """Test DC server/non-server capabilities"""
    print("\n" + "=" * 60)
    print("Test 4: DC Capabilities")
    print("=" * 60)
    
    try:
        from DRL.core.dc import DataCenter
        
        # Create server DC
        server_dc = DataCenter(0, cpu=100, ram=256, storage=1024, delay=5.0, is_server=True)
        print(f"✓ Server DC created:")
        print(f"    - Can host VNFs: {server_dc.is_server}")
        print(f"    - Resources: CPU={server_dc.cpu}, RAM={server_dc.ram}")
        print(f"    - Has resources for VNF 0: {server_dc.has_resources(0)}")
        
        # Create non-server DC
        non_server_dc = DataCenter(1, is_server=False)
        print(f"\n✓ Non-server DC created:")
        print(f"    - Can host VNFs: {non_server_dc.is_server}")
        print(f"    - Resources: CPU={non_server_dc.cpu}, RAM={non_server_dc.ram}")
        print(f"    - Has resources for VNF 0: {non_server_dc.has_resources(0)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed DC capability test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DRL-NFV Environment Test Suite")
    print("=" * 60)
    
    # Test 1: Data loading
    dc_list, topology, vnf_specs, requests_data = test_data_loading()
    if dc_list is None:
        print("\n✗ Test suite failed at data loading")
        return
    
    # Test 4: DC capabilities
    test_dc_capabilities()
    
    # Test 2: Environment initialization
    env = test_environment_initialization(dc_list, topology, requests_data)
    if env is None:
        print("\n✗ Test suite failed at environment initialization")
        return
    
    # Test 3: Environment step
    success = test_environment_step(env)
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests completed successfully!")
    else:
        print("✗ Some tests failed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
