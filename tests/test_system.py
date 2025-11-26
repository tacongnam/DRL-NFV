import unittest
import numpy as np
from env.core_network import CoreNetworkEnv
from models.vae_model import GenAIAgent
from models.dqn_model import DQNAgent
from config import *

class TestSystem(unittest.TestCase):
    def setUp(self):
        self.env = CoreNetworkEnv()
        self.vae = GenAIAgent()
        self.dqn = DQNAgent()
        
    def test_env_dimensions(self):
        dc_states = self.env._get_dc_states()
        self.assertEqual(dc_states.shape, (NUM_DCS, STATE_DIM_DC))
        
    def test_vae_prediction(self):
        dc_states = self.env._get_dc_states()
        values = self.vae.predict_value(dc_states)
        self.assertEqual(values.shape, (NUM_DCS, 1))
        
    def test_dqn_act(self):
        idx = 0
        dc_s, loc_s, glob_s = self.env.get_dqn_state(idx)
        action = self.dqn.act(dc_s, loc_s, glob_s)
        self.assertTrue(0 <= action < ACTION_SPACE)
        
    def test_full_step_cycle(self):
        self.env.reset()
        # VAE Select
        dc_states = self.env._get_dc_states()
        idx = np.argmax(self.vae.predict_value(dc_states))
        
        # DQN Act
        dc_s, loc_s, glob_s = self.env.get_dqn_state(idx)
        action = self.dqn.act(dc_s, loc_s, glob_s)
        
        # Step
        _, reward, _, _ = self.env.step(idx, action)
        self.assertIsInstance(reward, float)

if __name__ == '__main__':
    unittest.main()