import gymnasium as gym
import numpy as np
import config
from core import DataCenter, SwitchNode, TopologyManager, SFCManager, Simulator

class SFCEnvironment(gym.Env):
    def __init__(self, physical_graph, dcs_data, requests_data, dc_selector=None):
        self.topology = TopologyManager(physical_graph.copy(), k_paths=3)
        self.initial_dcs = dcs_data
        self.requests_data = requests_data
        
        self.sfc_manager = None
        self.simulator = None
        self.dcs = []
        self.dc_selector = dc_selector
        
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        chain_feat = 4 + config.MAX_VNF_TYPES + 3
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-1, high=np.inf, shape=(2 + 2*config.MAX_VNF_TYPES,), dtype=np.float32),
            gym.spaces.Box(low=-1, high=np.inf, shape=(config.MAX_VNF_TYPES + 3*chain_feat,), dtype=np.float32),
            gym.spaces.Box(low=-1, high=np.inf, shape=(4 + config.MAX_VNF_TYPES + 5*chain_feat,), dtype=np.float32)
        ))
        
        self.dc_order = []
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.step_count = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.topology = TopologyManager(self.topology.physical_graph.copy(), k_paths=3)
        self.dcs = [self._create_dc(dc) for dc in self.initial_dcs]
        
        self.sfc_manager = SFCManager()
        self.sfc_manager.load(self.requests_data)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        
        self.simulator.reset()
        self.sfc_manager.activate_new_requests(0)
        
        self._update_dc_order()
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        if not self.dc_order:
            self._update_dc_order()
        
        curr_dc_id = self.dc_order[self.current_dc_idx]
        curr_dc = self._get_dc_by_id(curr_dc_id)
        
        reward, completed = self._execute_action(curr_dc, action)
        
        server_count = sum(1 for dc in self.dcs if dc.is_server)
        self.current_dc_idx = (self.current_dc_idx + 1) % max(1, server_count)
        self.actions_this_step += 1
        
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            self._update_dc_order()
            self.current_dc_idx = 0
        
        done = self.simulator.is_done()
        self.step_count += 1
        
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'avg_e2e_delay': stats['avg_e2e_delay'],
            'total_generated': stats['total_generated'],
            'action_mask': self._get_valid_actions_mask(),
            'step': self.step_count
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_action(self, dc, action):
        from envs.action_handler import ActionHandler
        return ActionHandler.execute(self, dc, action)
    
    def _update_dc_order(self):
        server_dcs = [dc for dc in self.dcs if dc.is_server]
        if self.dc_selector is not None:
            self.dc_order = self.dc_selector.get_dc_order(
                server_dcs, 
                self.sfc_manager.active_requests,
                self.topology
            )
        else:
            self.dc_order = [dc.id for dc in server_dcs]
            np.random.shuffle(self.dc_order)
    
    def _get_observation(self):
        from envs.observer import Observer
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self._get_dc_by_id(self.dc_order[self.current_dc_idx])
        return Observer.get_drl_observation(curr_dc, self.sfc_manager)
    
    def _get_valid_actions_mask(self):
        from envs.utils import get_valid_actions_mask
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self._get_dc_by_id(self.dc_order[self.current_dc_idx])
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)
    
    def _create_dc(self, dc_config):
        if dc_config.is_server:
            return DataCenter(
                dc_config.id,
                cpu=dc_config.cpu,
                ram=dc_config.ram,
                storage=dc_config.storage,
                delay=dc_config.delay,
                cost_c=dc_config.cost_c,
                cost_h=dc_config.cost_h,
                cost_r=dc_config.cost_r
            )
        else:
            return SwitchNode(dc_config.id)
    
    def _get_dc_by_id(self, dc_id):
        for dc in self.dcs:
            if dc.id == dc_id:
                return dc
        return self.dcs[0]
    
    def get_first_dc(self):
        server_dc = next((d for d in self.dcs if d.is_server), None)
        if server_dc is None:
            raise ValueError("No server DCs found to determine VAE state dimension.")
        else:
            return server_dc