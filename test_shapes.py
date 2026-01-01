"""
Test script to verify observation dimensions
Run: python test_shapes.py
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import config
from runners.read_data import Read_data
from envs.drl_env import DRLEnv
from envs.observer import Observer

def test_dimensions():
    print("="*80)
    print("TESTING OBSERVATION DIMENSIONS")
    print("="*80)
    
    # Load test data
    reader = Read_data('data/test.json')
    graph = reader.get_G()
    dc_list = reader.get_V()
    vnf_specs = reader.get_F()
    requests_data = reader.get_R()
    
    config.update_vnf_specs(vnf_specs)
    config.ACTION_SPACE_SIZE = config.get_action_space_size()
    
    print(f"\nDataset Info:")
    print(f"  VNF Types: {config.NUM_VNF_TYPES}")
    print(f"  Action Space: {config.ACTION_SPACE_SIZE}")
    
    # Create environment
    env = DRLEnv(graph=graph, dcs=dc_list, requests_data=requests_data)
    state, _ = env.reset()
    
    print(f"\nObservation Shapes:")
    print(f"  s1 (DC State):       {state[0].shape}")
    print(f"  s2 (DC Demand):      {state[1].shape}")
    print(f"  s3 (Global State):   {state[2].shape}")
    
    # Expected dimensions
    V = config.NUM_VNF_TYPES
    chain_feature_size = 4 + V + 3
    
    expected_s1 = 2 + 2*V
    expected_s2 = V + 3*chain_feature_size
    expected_s3 = 4 + V + 5*chain_feature_size
    
    print(f"\nExpected Shapes (V={V}):")
    print(f"  s1: {expected_s1} = 2 + 2×{V}")
    print(f"  s2: {expected_s2} = {V} + 3×({chain_feature_size})")
    print(f"  s3: {expected_s3} = 4 + {V} + 5×({chain_feature_size})")
    
    total_expected = expected_s1 + expected_s2 + expected_s3
    total_actual = state[0].shape[0] + state[1].shape[0] + state[2].shape[0]
    
    print(f"\nTotal Features:")
    print(f"  Expected: {total_expected}")
    print(f"  Actual:   {total_actual}")
    
    # Verify
    assert state[0].shape[0] == expected_s1, f"s1 mismatch: {state[0].shape[0]} != {expected_s1}"
    assert state[1].shape[0] == expected_s2, f"s2 mismatch: {state[1].shape[0]} != {expected_s2}"
    assert state[2].shape[0] == expected_s3, f"s3 mismatch: {state[2].shape[0]} != {expected_s3}"
    
    print("\n✓ All dimensions match!")
    
    # Test VAE state
    server_dcs = [dc for dc in env.dcs if dc.is_server]
    if server_dcs:
        vae_states = Observer.get_all_dc_states(server_dcs, env.sfc_manager)
        vae_state_dim = Observer.get_state_dim()
        
        print(f"\nVAE State:")
        print(f"  Expected dim: {vae_state_dim}")
        print(f"  Actual shape: {vae_states.shape}")
        print(f"  Per DC: {vae_states.shape[1] if len(vae_states.shape) > 1 else vae_states.shape[0]}")
        
        assert vae_states.shape[1] == vae_state_dim, f"VAE state mismatch"
        print("✓ VAE dimensions match!")
    
    # Test chain pattern encoding
    print(f"\nChain Pattern Encoding:")
    test_chain = [0, 1]  # NAT -> FW
    encoded = Observer._encode_chain_pattern(test_chain, max_length=4)
    print(f"  Chain {test_chain} encoded shape: {encoded.shape}")
    print(f"  Expected: {chain_feature_size}")
    print(f"  Sequence: {encoded[:4]}")
    print(f"  Presence: {encoded[4:4+V]}")
    print(f"  Stats placeholder: {encoded[-3:]}")
    
    assert encoded.shape[0] == chain_feature_size, "Chain encoding mismatch"
    print("✓ Chain encoding correct!")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)

if __name__ == "__main__":
    test_dimensions()