# environment/gym_env.py
import gymnasium as gym
import numpy as np
import config
from spaces.sfc_manager import SFC_Manager
from spaces.dc import DataCenter
from spaces.topology import TopologyManager
from environment.observer import Observer
from environment.controller import ActionController
from environment.simulator import Simulator
from environment.priority import PriorityManager
from environment.utils import get_valid_actions_mask

class Env(gym.Env):
    """Custom Gymnasium Environment cho SFC Provisioning"""
    
    def __init__(self):
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        # Observation space: 3 inputs
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(2 * config.NUM_VNF_TYPES + 2,), 
                          dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),), 
                          dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),), 
                          dtype=np.float32)
        ))
        
        # Components
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.topology = None
        self.simulator = None
        self.controller = None
        self.priority_manager = None

        self.count_step = 0
        
        # Episode state
        self.current_dc_idx = 0
        self.dc_priority_order = []
        self.actions_this_step = 0
        
    def reset(self, num_dcs=None, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # 1. Setup DCs
        n = num_dcs if num_dcs else np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(n)]
        
        # 2. Setup topology
        self.topology = TopologyManager(n)
        
        # 3. Setup managers
        self.sfc_manager = SFC_Manager()
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        self.priority_manager = PriorityManager(self.topology)
        
        self.simulator.reset()
        
        # 4. Generate initial traffic
        if self.simulator.should_generate_initial_traffic():
            self.sfc_manager.generate_requests(0, len(self.dcs))
            self.simulator.has_generated_initial = True
        
        # 5. Initialize DC priority order
        self._update_dc_priority_order()
        self.current_dc_idx = 0
        self.actions_this_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Lấy observation từ DC hiện tại"""
        if not self.dc_priority_order:
            self._update_dc_priority_order()
        
        curr_dc = self.dcs[self.dc_priority_order[self.current_dc_idx]]
        return Observer.get_full_state(curr_dc, self.sfc_manager)

    def _get_valid_actions_mask(self):
        """Lấy action mask cho DC hiện tại"""
        curr_dc = self.dcs[self.dc_priority_order[self.current_dc_idx]]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)

    def _update_dc_priority_order(self):
        """Cập nhật priority order của DCs"""
        self.dc_priority_order = self.priority_manager.get_dc_priority_order(
            self.dcs, 
            self.sfc_manager.active_requests
        )

    def step(self, action):
        """
        Thực hiện một step
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current DC
        curr_dc_id = self.dc_priority_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        # 1. Execute action
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        # 2. Move to next DC in priority order
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        
        # 3. Count actions in this time step
        self.actions_this_step += 1
        
        # 4. Advance time after A actions
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            # Reset counter và update priority
            self.actions_this_step = 0
            self._update_dc_priority_order()
            self.current_dc_idx = 0
        
        # 5. Check done
        done = self.simulator.is_done()
        
        # 6. Get info
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay']
        }
        
        return self._get_obs(), reward, done, False, info