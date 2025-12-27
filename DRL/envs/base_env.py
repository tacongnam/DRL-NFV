import gymnasium as gym
import numpy as np
from DRL import config
from DRL.core.sfc_manager import SFC_Manager
from DRL.core.dc import DataCenter
from DRL.core.topology import TopologyManager
from DRL.envs.observer import Observer
from DRL.envs.controller import ActionController
from DRL.core.simulator import Simulator
from DRL.envs.utils import get_valid_actions_mask

class SFCBaseEnv(gym.Env):
    """
    Base Environment for SFC Provisioning.
    Contains common logic for both pure DRL and VAE-assisted DRL.
    """
    
    def __init__(self, dcs=None, topology=None, requests_data=None):
        """
        Initialize environment with loaded data.
        
        Args:
            dcs: List of DataCenter objects from Read_data.get_V()
            topology: TopologyManager from Read_data.get_E()
            requests_data: List of request dicts from Read_data.get_R()
        """
        super().__init__()
        
        # Store loaded data
        self.initial_dcs = dcs
        self.initial_topology = topology
        self.requests_data = requests_data
        
        # --- 1. Define Spaces (Common) ---
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        # Note: Observation space structure depends on NUM_SFC_TYPES which is deprecated
        # For now, keep it for backward compatibility, but it won't be used with loaded data
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
        
        # --- 2. Core Components (Placeholder) ---
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.topology = None
        self.simulator = None
        self.controller = None
        
        # --- 3. State Variables ---
        self.count_step = 0
        self.current_dc_idx = 0
        self.dc_order = []  # List containing DC traversal order (VAE or Priority decides)
        self.actions_this_step = 0

    def reset(self, seed=None):
        """Reset environment logic"""
        super().reset(seed=seed)
        
        # Use loaded data or create default
        if self.initial_dcs is not None:
            # Deep copy DCs to reset resources
            self.dcs = []
            for dc in self.initial_dcs:
                if dc.is_server:
                    new_dc = DataCenter(
                        dc.id, cpu=dc.cpu, ram=dc.ram, storage=dc.storage,
                        delay=dc.delay, cost_c=dc.cost_c, cost_h=dc.cost_h,
                        cost_r=dc.cost_r, is_server=True
                    )
                else:
                    new_dc = DataCenter(dc.id, is_server=False)
                self.dcs.append(new_dc)
            
            self.topology = self.initial_topology
        else:
            # Fallback: create random topology (for testing without data)
            n = np.random.randint(2, config.MAX_NUM_DCS + 1)
            self.dcs = [DataCenter(i, is_server=True) for i in range(n)]
            self.topology = TopologyManager(n)
        
        # Setup managers
        self.sfc_manager = SFC_Manager()
        
        # Load requests if provided
        if self.requests_data is not None:
            self.sfc_manager.load_requests(self.requests_data)
        
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        
        # Hook for child class to initialize more (e.g., PriorityManager)
        self._after_init_components() 
        
        self.simulator.reset()
        
        # Activate initial requests (those with arrival_time = 0)
        self.sfc_manager.activate_requests_at_time(0)
        
        # Reset state logic
        self._update_dc_order()  # Child class will define this
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.count_step = 0
        
        # Hook for specific state reset (e.g., clear prev_states for VAE)
        self._reset_specific_state()
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Common step logic.
        Uses Template Method Pattern with hooks.
        """
        # Hook: Before executing action (VAE uses to snapshot state)
        self._pre_action_hook()
        
        # Get current DC
        if not self.dc_order:
             self._update_dc_order()
             
        curr_dc_id = self.dc_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        # Execute action
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        # Move to next DC
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        self.actions_this_step += 1
        
        # Advance time logic
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            # Update DC order (Random/VAE or Priority)
            self._update_dc_order()
            self.current_dc_idx = 0
        
        # Check done
        done = self.simulator.is_done()
        self.count_step += 1
        
        # Info
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay']
        }
        
        # Hook: Add specific info (VAE adds prev_states)
        self._post_step_info_hook(info)
        
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """Get observation based on current dc_order"""    
        if not self.dc_order:
            self._update_dc_order()
            
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        
        # Call new observation function
        return Observer.get_drl_observation(curr_dc, self.sfc_manager)

    def _get_valid_actions_mask(self):
        """Get mask based on current dc_order"""
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)

    # --- Abstract Methods / Hooks (Child class needs to override) ---
    def _update_dc_order(self):
        """Determine DC traversal order (VAE vs Priority)"""
        raise NotImplementedError

    def _after_init_components(self):
        """Initialize additional managers if needed"""
        pass

    def _reset_specific_state(self):
        """Reset specific state variables"""
        pass

    def _pre_action_hook(self):
        """Run before executing action"""
        pass
        
    def _post_step_info_hook(self, info):
        """Add data to info"""
        pass