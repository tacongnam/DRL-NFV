import gymnasium as gym
import numpy as np
import config
from core.sfc_manager import SFC_Manager
from core.dc import DataCenter
from core.topology import TopologyManager
from envs.observer import Observer
from envs.controller import ActionController
from core.simulator import Simulator
from envs.utils import get_valid_actions_mask
import networkx as nx

class SFCBaseEnv(gym.Env):
    """Base Environment with graph-based routing"""
    
    def __init__(self, graph=None, dcs=None, requests_data=None):
        """
        Initialize with physical network graph.
        
        Args:
            graph: NetworkX graph from Read_data.get_G()
            dcs: List of DataCenter objects from Read_data.get_V()
            requests_data: List of request dicts from Read_data.get_R()
        """
        super().__init__()
        
        self.initial_graph = graph
        self.initial_dcs = dcs
        self.requests_data = requests_data
        
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
        
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.topology = None
        self.simulator = None
        self.controller = None
        
        self.count_step = 0
        self.current_dc_idx = 0
        self.dc_order = []
        self.actions_this_step = 0

    def reset(self, seed=None, num_dcs=None):
        """Reset environment with graph"""
        super().reset(seed=seed)
        
        if self.initial_graph is not None and self.initial_dcs is not None:
            # Deep copy graph to reset bandwidth
            self.topology = TopologyManager(self.initial_graph.copy())
            
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
        else:
            # Fallback: create random topology (for testing)
            n = num_dcs if num_dcs is not None else np.random.randint(2, config.MAX_NUM_DCS + 1)
            self.dcs = [DataCenter(i, is_server=True) for i in range(n)]
            
            # Create simple graph
            G = nx.Graph()
            for i in range(n):
                G.add_node(i, server=True, cpu=config.DC_CPU_CYCLES, 
                          ram=config.DC_RAM, storage=config.DC_STORAGE, delay=1.0)
            
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(i, j, bw=config.LINK_BW_CAPACITY, 
                              delay=np.random.uniform(1, 10), 
                              capacity=config.LINK_BW_CAPACITY)
            
            self.topology = TopologyManager(G)
        
        self.sfc_manager = SFC_Manager()
        
        if self.requests_data is not None:
            self.sfc_manager.load_requests(self.requests_data)
        
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        
        self._after_init_components() 
        
        self.simulator.reset()
        self.sfc_manager.activate_requests_at_time(0)
        
        self._update_dc_order()
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.count_step = 0
        
        self._reset_specific_state()
        
        return self._get_obs(), {}

    def step(self, action):
        """Step with routing-aware rewards"""
        self._pre_action_hook()
        
        if not self.dc_order:
             self._update_dc_order()
             
        curr_dc_id = self.dc_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        self.actions_this_step += 1
        
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            self._update_dc_order()
            self.current_dc_idx = 0
        
        done = self.simulator.is_done()
        self.count_step += 1
        
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay']
        }
        
        self._post_step_info_hook(info)
        
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        if not self.dc_order:
            self._update_dc_order()
            
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return Observer.get_drl_observation(curr_dc, self.sfc_manager)

    def _get_valid_actions_mask(self):
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)

    def _update_dc_order(self):
        raise NotImplementedError

    def _after_init_components(self):
        pass

    def _reset_specific_state(self):
        pass

    def _pre_action_hook(self):
        pass
        
    def _post_step_info_hook(self, info):
        pass