"""
Simple test script to verify data loading without dependencies on gymnasium.
"""
import numpy as np
from data_info.read_data import Read_data

def test_data_loading():
    """Test loading data from JSON file"""
    print("=" * 60)
    print("Test: Data Loading")
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
        for i, dc in enumerate(dc_list[:5]):
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
            startup_items = list(specs['startup_time'].items())[:3]
            print(f"      Startup times: {startup_items}...")
        
        # Show sample request info
        print(f"\n  Sample Request info:")
        for req in requests_data[:5]:
            print(f"    Request {req['id']}: T={req['arrival_time']}, "
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
        
        # Test topology access
        print(f"\n  Testing Topology:")
        print(f"    - Delay matrix shape: {topology.delay_matrix.shape}")
        print(f"    - Bandwidth matrix shape: {topology.bw_matrix.shape}")
        
        # Show some connections
        print(f"    - Sample delays (DC 0 to others):")
        for j in range(min(5, len(dc_list))):
            delay = topology.get_propagation_delay(0, j)
            bw = topology.bw_matrix[0, j]
            if delay == np.inf:
                print(f"      DC 0 -> DC {j}: No connection (delay=inf, bw={bw:.0f})")
            else:
                print(f"      DC 0 -> DC {j}: delay={delay:.2f}ms, bw={bw:.0f}")
        
        # Test DC capabilities
        print(f"\n  Testing DC Capabilities:")
        for dc in dc_list[:3]:
            print(f"    DC {dc.id}:")
            print(f"      - Is server: {dc.is_server}")
            if dc.is_server:
                print(f"      - Can host VNF 0: {dc.has_resources(0)}")
                print(f"      - Resources: CPU={dc.cpu}, RAM={dc.ram}, Storage={dc.storage}")
            else:
                print(f"      - Cannot host VNFs")
        
        # Test request data structure
        print(f"\n  Testing Request Data Structure:")
        if requests_data:
            req = requests_data[0]
            print(f"    Request 0 fields:")
            for key, value in req.items():
                print(f"      - {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✓ All data loading tests passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run test"""
    print("\n" + "=" * 60)
    print("DRL-NFV Data Loading Test")
    print("=" * 60 + "\n")
    
    success = test_data_loading()
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed!")
    

if __name__ == "__main__":
    main()
