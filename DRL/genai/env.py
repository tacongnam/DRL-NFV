import gymnasium as gym
import numpy as np
import config
from core.dc import DataCenter, SwitchNode
from core.topology import TopologyManager
from core.sfc_manager import SFCManager
from core.simulator import Simulator
from core.vnf import VNFInstance

class SFCEnvironment(gym.Env):
    """
    Unified RL Environment for SFC Placement
    """
    
    def __init__(self, graph, dcs_data, requests_data, dc_selector=None):
        super().__init__()
        
        # Infrastructure
        # graph is physical graph (including switches)
        self.topology = TopologyManager(graph.copy(), k_paths=3)
        self.initial_dcs = dcs_data
        self.requests_data = requests_data
        
        # Components
        self.sfc_manager = None
        self.simulator = None
        self.dcs = []
        self.dc_selector = dc_selector
        
        # Action space
        V = config.NUM_VNF_TYPES
        self.action_space = gym.spaces.Discrete(2 * V + 1)
        
        # Observation space (defined by Observer)
        chain_feat = 4 + V + 3
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=-1, high=np.inf, shape=(2 + 2*V,), dtype=np.float32),
            gym.spaces.Box(low=-1, high=np.inf, shape=(V + 3*chain_feat,), dtype=np.float32),
            gym.spaces.Box(low=-1, high=np.inf, shape=(4 + V + 5*chain_feat,), dtype=np.float32)
        ))
        
        # Episode state
        self.dc_order = []
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.step_count = 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset infrastructure
        self.topology = TopologyManager(self.topology.physical_graph.copy(), k_paths=3)
        self.dcs = [self._create_dc(dc) for dc in self.initial_dcs]
        
        # Reset managers
        self.sfc_manager = SFCManager()
        self.sfc_manager.load(self.requests_data)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        
        # Activate initial requests
        self.simulator.reset()
        self.sfc_manager.activate_new_requests(0)
        
        # Initialize episode
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
        
        # Execute action
        reward, completed = self._execute_action(curr_dc, action)
        
        # Move to next DC
        server_count = sum(1 for dc in self.dcs if dc.is_server)
        self.current_dc_idx = (self.current_dc_idx + 1) % max(1, server_count)
        self.actions_this_step += 1
        
        # Check if round complete
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
        if action == 0: # WAIT
            return config.REWARD_WAIT, False
        
        vnf_type = (action - 1) % config.NUM_VNF_TYPES
        is_allocate = action > config.NUM_VNF_TYPES
        
        if is_allocate:
            return self._handle_allocation(dc, vnf_type)
        else:
            return self._handle_uninstall(dc, vnf_type)
    
    def _handle_allocation(self, dc, vnf_type):
        # Find best request
        req = self._select_best_request(vnf_type, dc.id)
        if req is None:
            return config.REWARD_INVALID, False
        
        # 1. Get or Install VNF
        # Check for idle VNF first
        vnf = dc._get_idle_vnf(vnf_type)
        installed_new = False
        
        if vnf is None:
            # Try to install new VNF
            vnf = dc.install_vnf(vnf_type) # This handles resource check and consumption
            if vnf is None:
                return config.REWARD_INVALID, False
            installed_new = True
        
        # 2. Route Bandwidth (if needed)
        last_dc = req.get_last_placed_dc()
        prop_delay = 0.0
        hop_count = 0
        
        if last_dc != dc.id:
            # Use TopologyManager to allocate BW on physical links
            success, path, delay = self.topology.allocate_bandwidth(
                last_dc, dc.id, req.bandwidth
            )
            
            if not success:
                # Rollback if VNF was just installed
                if installed_new:
                    dc.uninstall_vnf(vnf_type)
                return config.REWARD_INVALID, False
            
            prop_delay = delay
            hop_count = len(path) - 1
            req.allocated_paths.append((last_dc, dc.id, req.bandwidth, path))
        
        # 3. Assign VNF to request
        dc_delay = dc.delay if dc.delay else 0.0
        vnf.assign(req.id, dc_delay, waiting_time=0.0)
        
        proc_delay = vnf.get_processing_time(dc_delay)
        req.advance_chain(dc.id, prop_delay, proc_delay, vnf)
        req.check_completion()
        
        # 4. Calculate Reward
        base_reward = config.REWARD_SATISFIED if req.is_completed else config.REWARD_SATISFIED / 2
        delay_penalty = config.ALPHA_DELAY_PENALTY * prop_delay
        hop_penalty = config.BETA_HOP_PENALTY * hop_count
        reward = base_reward - delay_penalty - hop_penalty
        
        # 5. Cleanup if completed
        if req.is_completed:
            for _, _, bw, path in req.allocated_paths:
                self.topology.release_bandwidth_on_path(path, bw)
            req.allocated_paths.clear()
        
        return reward, req.is_completed
    
    def _handle_uninstall(self, dc, vnf_type):
        # Check if needed by any active request
        is_needed = any(
            r.get_next_vnf() == vnf_type 
            for r in self.sfc_manager.active_requests
        )
        if is_needed:
            return config.REWARD_UNINSTALL_NEEDED, False
        
        # Try to uninstall
        success = dc.uninstall_vnf(vnf_type)
        if success:
            return 0.0, False
        
        return config.REWARD_INVALID, False
    
    def _select_best_request(self, vnf_type, dc_id):
        candidates = []
        for req in self.sfc_manager.active_requests:
            if (not req.is_completed and not req.is_dropped and 
                req.get_next_vnf() == vnf_type):
                priority = self._calculate_priority(req, dc_id)
                candidates.append((priority, req))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def _calculate_priority(self, req, dc_id):
        p1 = req.elapsed_time - req.max_delay
        last_dc = req.get_last_placed_dc()
        p2 = 0.0
        if last_dc is not None:
            p2 = (config.PRIORITY_P2_SAME_DC if last_dc == dc_id 
                  else config.PRIORITY_P2_DIFF_DC)
        remaining = req.get_remaining_time()
        p3 = 0.0
        if remaining < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining + config.EPSILON_SMALL)
        return p1 + p2 + p3
    
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