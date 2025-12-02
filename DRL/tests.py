import unittest
import numpy as np
from env.sfc_environment import SFCEnvironment
from agent import DQNAgent
from config import *
from utils import *

class TestSFCEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = SFCEnvironment(num_dcs=4)
    
    def test_environment_initialization(self):
        self.assertEqual(len(self.env.dcs), 4)
        self.assertEqual(len(self.env.pending_sfcs), 0)
        
        for dc in self.env.dcs:
            self.assertGreaterEqual(dc['cpu'], DC_CONFIG['cpu_range'][0])
            self.assertLessEqual(dc['cpu'], DC_CONFIG['cpu_range'][1])
            self.assertEqual(dc['ram'], DC_CONFIG['ram'])
            self.assertEqual(dc['storage'], DC_CONFIG['storage'])
    
    def test_sfc_generation(self):
        self.env._generate_sfc_requests()
        self.assertGreater(len(self.env.pending_sfcs), 0)
        
        for sfc in self.env.pending_sfcs:
            self.assertIn(sfc['type'], SFC_TYPES.keys())
            self.assertIn('vnfs', sfc)
            self.assertIn('bw', sfc)
            self.assertIn('delay', sfc)
    
    def test_observation_space(self):
        obs, _ = self.env.reset()
        
        self.assertIn('state1', obs)
        self.assertIn('state2', obs)
        self.assertIn('state3', obs)
        
        self.assertEqual(len(obs['state1']), 2 * len(VNF_LIST) + 2)
        self.assertEqual(len(obs['state2']), len(SFC_TYPES) * (1 + 2 * len(VNF_LIST)))
        self.assertEqual(len(obs['state3']), len(SFC_TYPES) * (4 + len(VNF_LIST)))
    
    def test_action_space(self):
        self.assertEqual(self.env.action_space.n, 2 * len(VNF_LIST) + 1)
    
    def test_step_function(self):
        obs, _ = self.env.reset()
        self.env._generate_sfc_requests()
        
        action = self.env.action_space.sample()
        next_obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertIn('state1', next_obs)
        self.assertIn('state2', next_obs)
        self.assertIn('state3', next_obs)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIn('acceptance_ratio', info)
    
    def test_vnf_allocation(self):
        self.env.reset()
        self.env._generate_sfc_requests()
        
        if self.env.pending_sfcs:
            sfc = self.env.pending_sfcs[0]
            vnf_type = sfc['vnfs'][0]
            vnf_idx = VNF_LIST.index(vnf_type)
            
            action = len(VNF_LIST) + vnf_idx
            obs, reward, done, _, info = self.env.step(action)
            
            self.assertTrue(reward >= REWARD_CONFIG['invalid_action'])

class TestUtils(unittest.TestCase):
    def test_network_topology_creation(self):
        G = create_network_topology(4)
        self.assertEqual(len(G.nodes), 4)
        self.assertGreater(len(G.edges), 0)
    
    def test_shortest_path(self):
        G = create_network_topology(4)
        path = get_shortest_path_with_bw(G, 0, 3, 100)
        
        if path:
            self.assertEqual(path[0], 0)
            self.assertEqual(path[-1], 3)
    
    def test_priority_calculations(self):
        p1 = calculate_priority_p1(30, 50)
        self.assertEqual(p1, -20)
        
        p3 = calculate_priority_p3(45, 50)
        self.assertGreater(p3, 0)
    
    def test_resource_availability(self):
        dc_resources = {'cpu': 100, 'ram': 256, 'storage': 2000}
        
        available = check_resource_availability(dc_resources, 'NAT', VNF_SPECS)
        self.assertTrue(available)
        
        dc_resources['cpu'] = 1
        available = check_resource_availability(dc_resources, 'NAT', VNF_SPECS)
        self.assertFalse(available)
    
    def test_replay_buffer(self):
        buffer = ReplayBuffer(100)
        
        for i in range(10):
            s1 = np.random.rand(10)
            s2 = np.random.rand(20)
            s3 = np.random.rand(30)
            a = i % 5
            r = np.random.rand()
            ns1 = np.random.rand(10)
            ns2 = np.random.rand(20)
            ns3 = np.random.rand(30)
            d = False
            
            buffer.push(s1, s2, s3, a, r, ns1, ns2, ns3, d)
        
        self.assertEqual(len(buffer), 10)
        
        batch = buffer.sample(5)
        self.assertEqual(len(batch), 9)

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state1_dim = 2 * len(VNF_LIST) + 2
        self.state2_dim = len(SFC_TYPES) * (1 + 2 * len(VNF_LIST))
        self.state3_dim = len(SFC_TYPES) * (4 + len(VNF_LIST))
        self.action_dim = 2 * len(VNF_LIST) + 1
        
        self.agent = DQNAgent(self.state1_dim, self.state2_dim, 
                             self.state3_dim, self.action_dim)
    
    def test_agent_initialization(self):
        self.assertEqual(self.agent.action_dim, 2 * len(VNF_LIST) + 1)
        self.assertEqual(self.agent.epsilon, DRL_CONFIG['epsilon_start'])
    
    def test_action_selection(self):
        state = {
            'state1': np.random.rand(self.state1_dim).astype(np.float32),
            'state2': np.random.rand(self.state2_dim).astype(np.float32),
            'state3': np.random.rand(self.state3_dim).astype(np.float32)
        }
        
        action = self.agent.select_action(state, training=False)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_epsilon_decay(self):
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
    
    def test_memory_storage(self):
        state = {
            'state1': np.random.rand(self.state1_dim).astype(np.float32),
            'state2': np.random.rand(self.state2_dim).astype(np.float32),
            'state3': np.random.rand(self.state3_dim).astype(np.float32)
        }
        
        next_state = {
            'state1': np.random.rand(self.state1_dim).astype(np.float32),
            'state2': np.random.rand(self.state2_dim).astype(np.float32),
            'state3': np.random.rand(self.state3_dim).astype(np.float32)
        }
        
        self.agent.store_transition(state, 0, 1.0, next_state, False)
        self.assertEqual(len(self.agent.memory), 1)

class TestConfig(unittest.TestCase):
    def test_sfc_types(self):
        self.assertEqual(len(SFC_TYPES), 6)
        self.assertIn('CG', SFC_TYPES)
        self.assertIn('AR', SFC_TYPES)
        
        for sfc_type, config in SFC_TYPES.items():
            self.assertIn('vnfs', config)
            self.assertIn('bw', config)
            self.assertIn('delay', config)
            self.assertIn('bundle', config)
    
    def test_vnf_specs(self):
        self.assertEqual(len(VNF_SPECS), 6)
        
        for vnf, specs in VNF_SPECS.items():
            self.assertIn('cpu', specs)
            self.assertIn('ram', specs)
            self.assertIn('storage', specs)
            self.assertIn('proc_time', specs)
    
    def test_drl_config(self):
        self.assertEqual(DRL_CONFIG['updates'], 350)
        self.assertEqual(DRL_CONFIG['episodes_per_update'], 20)
        self.assertGreater(DRL_CONFIG['gamma'], 0)
        self.assertLessEqual(DRL_CONFIG['gamma'], 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)