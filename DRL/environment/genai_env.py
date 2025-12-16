import gymnasium as gym
import numpy as np
import config
from spaces.sfc_manager import SFC_Manager
from spaces.dc import DataCenter
from spaces.topology import TopologyManager
from environment.observer import Observer
from environment.controller import ActionController
from environment.simulator import Simulator
from environment.utils import get_valid_actions_mask
from genai.selector import GenAIDCSelector
from genai.observer import DCStateObserver

class GenAIEnv(gym.Env):
    """Environment với GenAI DC selection"""
    
    def __init__(self, genai_model=None, data_collection_mode=False):
        """
        Args:
            genai_model: GenAIModel instance (None = fallback to random)
            data_collection_mode: Nếu True, chọn DC random để thu thập data
        """
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
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
        
        # GenAI components
        self.genai_model = genai_model
        self.dc_selector = GenAIDCSelector(genai_model) if genai_model else None
        self.data_collection_mode = data_collection_mode
        
        self.count_step = 0
        
        # Episode state
        self.current_dc_idx = 0
        self.dc_order = []
        self.actions_this_step = 0
        
        # Data collection
        self.dc_prev_states = {}  # {dc_id: prev_state}
    
    def reset(self, num_dcs=None, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Setup DCs
        n = num_dcs if num_dcs else np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(n)]
        
        # Setup topology
        self.topology = TopologyManager(n)
        
        # Setup managers
        self.sfc_manager = SFC_Manager()
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        self.simulator.reset()
        
        # Generate initial traffic
        if self.simulator.should_generate_initial_traffic():
            self.sfc_manager.generate_requests(0, len(self.dcs))
            self.simulator.has_generated_initial = True
        
        # Initialize DC selection
        self._update_dc_order()
        self.current_dc_idx = 0
        self.actions_this_step = 0
        
        # Clear prev states
        self.dc_prev_states = {}
        
        return self._get_obs(), {}
    
    def _update_dc_order(self):
        """Cập nhật DC selection order"""
        if self.data_collection_mode or self.dc_selector is None:
            # Random selection for data collection
            self.dc_order = list(range(len(self.dcs)))
            np.random.shuffle(self.dc_order)
        else:
            # GenAI selection
            self.dc_order = self.dc_selector.get_dc_ranking(self.dcs, self.sfc_manager)
    
    def _get_obs(self):
        """Lấy observation từ DC hiện tại"""
        if not self.dc_order:
            self._update_dc_order()
        
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return Observer.get_full_state(curr_dc, self.sfc_manager)
    
    def _get_valid_actions_mask(self):
        """Lấy action mask cho DC hiện tại"""
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)
    
    def step(self, action):
        """Execute step"""
        # Get current DC
        curr_dc_id = self.dc_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        # Store prev state for data collection
        if self.data_collection_mode and curr_dc_id not in self.dc_prev_states:
            self.dc_prev_states[curr_dc_id] = DCStateObserver.get_dc_state(
                curr_dc, self.sfc_manager
            )
        
        # Execute action
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        # Move to next DC
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        self.actions_this_step += 1
        
        # Advance time after A actions
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            self._update_dc_order()
            self.current_dc_idx = 0
        
        # Check done
        done = self.simulator.is_done()
        
        # Info
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay'],
            'dc_prev_states': self.dc_prev_states  # For data collection
        }
        
        return self._get_obs(), reward, done, False, info
    
    def get_dc_transitions(self):
        """
        Lấy transitions cho GenAI training
        
        Returns:
            List[(dc_id, prev_state, curr_state, value)]
        """
        transitions = []
        
        for dc in self.dcs:
            if dc.id in self.dc_prev_states:
                prev_state = self.dc_prev_states[dc.id]
                curr_state = DCStateObserver.get_dc_state(dc, self.sfc_manager)
                value = DCStateObserver.calculate_dc_value(dc, self.sfc_manager, prev_state)
                
                transitions.append((dc.id, prev_state, curr_state, value))
        
        # Clear for next batch
        self.dc_prev_states = {}
        
        return transitions