import sys
sys.path.append('..')

import numpy as np
from env.core_network import CoreNetwork, DataCenter
from env.traffic_generator import TrafficGenerator, SFCRequest
from models.dqn_model import DQNModel
from models.vae_model import VAEModel
from config import VNF_TYPES, VNF_SPECS

def test_datacenter():
    print("Testing DataCenter...")
    dc = DataCenter(0, cpu=100, storage=2000, ram=256)
    
    assert dc.can_install('NAT'), "Should be able to install NAT"
    assert dc.install_vnf('NAT'), "NAT installation failed"
    assert dc.installed_vnfs['NAT'] == 1, "NAT count should be 1"
    
    assert dc.can_allocate('NAT', 'sfc_1'), "Should be able to allocate NAT"
    assert dc.allocate_vnf('NAT', 'sfc_1'), "NAT allocation failed"
    
    assert not dc.can_allocate('NAT', 'sfc_2'), "Should not allocate when no free instance"
    
    assert dc.install_vnf('NAT'), "Should install second NAT"
    assert dc.can_allocate('NAT', 'sfc_2'), "Should allocate after second install"
    
    state = dc.get_state()
    assert len(state) == 10, f"State dimension should be 10, got {len(state)}"
    
    print("✓ DataCenter tests passed")

def test_core_network():
    print("\nTesting CoreNetwork...")
    network = CoreNetwork(num_dcs=4)
    
    assert len(network.dcs) == 4, "Should have 4 DCs"
    assert network.link_bw.shape == (4, 4), "Link bandwidth matrix wrong shape"
    
    assert network.get_available_bw(0, 1) == 1000, "Initial bandwidth should be 1000"
    
    network.allocate_bw([0, 1, 2], 100)
    assert network.get_available_bw(0, 1) == 900, "After allocation should be 900"
    assert network.get_available_bw(1, 2) == 900, "After allocation should be 900"
    
    network.deallocate_bw([0, 1], 100)
    assert network.get_available_bw(0, 1) == 1000, "After deallocation should be 1000"
    
    network_state = network.get_network_state()
    assert len(network_state) > 0, "Network state should not be empty"
    
    print("✓ CoreNetwork tests passed")

def test_sfc_request():
    print("\nTesting SFCRequest...")
    sfc = SFCRequest(0, 'CG', src_dc=0, dst_dc=3)
    
    assert sfc.type == 'CG', "SFC type should be CG"
    assert len(sfc.chain) == 5, "CG chain should have 5 VNFs"
    assert sfc.get_current_vnf() == 'NAT', "First VNF should be NAT"
    
    sfc.advance_vnf(0, VNF_SPECS['NAT']['process_time'])
    assert sfc.get_current_vnf() == 'FW', "Second VNF should be FW"
    assert not sfc.is_complete(), "Should not be complete after 1 VNF"
    
    for vnf in sfc.chain[1:]:
        sfc.advance_vnf(1, VNF_SPECS[vnf]['process_time'])
    
    assert sfc.is_complete(), "Should be complete after all VNFs"
    
    state = sfc.get_state()
    assert len(state) == 11, f"SFC state dimension should be 11, got {len(state)}"
    
    print("✓ SFCRequest tests passed")

def test_traffic_generator():
    print("\nTesting TrafficGenerator...")
    traffic_gen = TrafficGenerator(num_dcs=4)
    
    new_sfcs = traffic_gen.generate_bundle(request_count=1)
    assert len(new_sfcs) > 0, "Should generate SFC requests"
    assert traffic_gen.get_active_count() > 0, "Should have active SFCs"
    
    initial_count = len(traffic_gen.active_sfcs)
    traffic_gen.active_sfcs[0].active = False
    traffic_gen.remove_completed()
    assert len(traffic_gen.active_sfcs) == initial_count - 1, "Should remove completed SFC"
    
    print("✓ TrafficGenerator tests passed")

def test_vae_model():
    print("\nTesting VAE Model...")
    state_dim = 10
    vae = VAEModel(state_dim, latent_dim=16)
    
    test_states = np.random.rand(32, state_dim).astype(np.float32)
    
    z, z_mean, z_log_var = vae.encode(test_states)
    assert z.shape == (32, 16), f"Latent shape should be (32, 16), got {z.shape}"
    
    reconstructed = vae.decode(z)
    assert reconstructed.shape == (32, state_dim), f"Reconstructed shape mismatch"
    
    values = vae.get_dc_values(test_states)
    assert values.shape == (32,), f"Values shape should be (32,), got {values.shape}"
    
    next_states = np.random.rand(32, state_dim).astype(np.float32)
    value_targets = np.random.rand(32, 1).astype(np.float32)
    
    loss, recon_loss, kl_loss, val_loss = vae.train_step(test_states, next_states, value_targets)
    assert loss.numpy() > 0, "Loss should be positive"
    
    print("✓ VAE Model tests passed")

def test_dqn_model():
    print("\nTesting DQN Model...")
    dc_state_dim = 10
    sfc_state_dim = 11
    network_state_dim = 56
    
    dqn = DQNModel(dc_state_dim, sfc_state_dim, network_state_dim)
    
    assert dqn.num_actions == 13, f"Should have 13 actions, got {dqn.num_actions}"
    
    dc_state = np.random.rand(dc_state_dim).astype(np.float32)
    sfc_state = np.random.rand(sfc_state_dim).astype(np.float32)
    network_state = np.random.rand(network_state_dim).astype(np.float32)
    
    action = dqn.get_action(dc_state, sfc_state, network_state)
    assert 0 <= action < dqn.num_actions, f"Action should be in valid range"
    
    valid_actions = [0, 1, 12]
    action = dqn.get_action(dc_state, sfc_state, network_state, valid_actions)
    assert action in valid_actions, f"Action should be in valid actions list"
    
    batch_size = 32
    states = (
        np.random.rand(batch_size, dc_state_dim).astype(np.float32),
        np.random.rand(batch_size, sfc_state_dim).astype(np.float32),
        np.random.rand(batch_size, network_state_dim).astype(np.float32)
    )
    actions = np.random.randint(0, dqn.num_actions, batch_size)
    rewards = np.random.rand(batch_size).astype(np.float32)
    next_states = (
        np.random.rand(batch_size, dc_state_dim).astype(np.float32),
        np.random.rand(batch_size, sfc_state_dim).astype(np.float32),
        np.random.rand(batch_size, network_state_dim).astype(np.float32)
    )
    dones = np.random.randint(0, 2, batch_size).astype(np.float32)
    
    loss = dqn.train_step(states, actions, rewards, next_states, dones)
    assert loss.numpy() >= 0, "Loss should be non-negative"
    
    initial_weights = dqn.target_model.get_weights()[0].copy()
    dqn.model.set_weights([w + 0.1 for w in dqn.model.get_weights()])
    dqn.update_target_model()
    updated_weights = dqn.target_model.get_weights()[0]
    
    assert not np.allclose(initial_weights, updated_weights), "Target model should update"
    
    print("✓ DQN Model tests passed")

def test_constraints():
    print("\nTesting VNF Placement Constraints...")
    network = CoreNetwork(num_dcs=4)
    sfc = SFCRequest(0, 'AR', src_dc=0, dst_dc=3)
    
    dc0 = network.dcs[0]
    dc0.install_vnf('NAT')
    dc0.allocate_vnf('NAT', sfc.id)
    
    assert not dc0.can_allocate('NAT', sfc.id), "Same SFC cannot allocate twice to same VNF instance"
    
    dc0.install_vnf('NAT')
    assert dc0.can_allocate('NAT', 999), "Different SFC can use second instance"
    
    initial_storage = dc0.storage
    dc0.install_vnf('FW')
    assert dc0.storage < initial_storage, "Storage should decrease after install"
    
    dc0.uninstall_vnf('FW')
    assert dc0.storage == initial_storage, "Storage should restore after uninstall"
    
    print("✓ Constraint tests passed")

def run_all_tests():
    print("="*60)
    print("Running VNF Placement System Tests")
    print("="*60)
    
    test_datacenter()
    test_core_network()
    test_sfc_request()
    test_traffic_generator()
    test_vae_model()
    test_dqn_model()
    test_constraints()
    
    print("\n" + "="*60)
    print("All tests passed successfully! ✓")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()