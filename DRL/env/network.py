import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import config
from env.utils import DataCenter, calculate_priority, VNFInstance
from env.sfc import SFCManager

class SFCNVEnv(gym.Env):
    def __init__(self):
        super(SFCNVEnv, self).__init__()
        
        self.action_space = spaces.Discrete(config.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=10000, shape=(2 * config.NUM_VNF_TYPES + 2,)),
            spaces.Box(low=0, high=10000, shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),)),
            spaces.Box(low=0, high=10000, shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),))
        ))
        
        self.sfc_manager = SFCManager()
        self.dcs = []
        self.topology = None
        self.num_active_dcs = config.MAX_NUM_DCS
        
        self.current_dc_idx = 0
        self.sim_time_ms = 0
        self.action_step_counter = 0
        
        self.total_generated_requests = 0
        self.total_accepted_requests = 0

    def reset(self, seed=None, num_dcs=None):
        super().reset(seed=seed)
        
        if num_dcs is not None:
            self.num_active_dcs = num_dcs
        else:
            self.num_active_dcs = np.random.randint(2, config.MAX_NUM_DCS + 1)

        self.dcs = [DataCenter(i) for i in range(self.num_active_dcs)]
        self.topology = nx.complete_graph(self.num_active_dcs)
        
        self.sfc_manager = SFCManager()
        self.sim_time_ms = 0
        self.action_step_counter = 0
        self.current_dc_idx = 0
        
        self.total_generated_requests = 0
        self.total_accepted_requests = 0
        
        generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
        self.total_generated_requests += generated
        
        return self._get_state(), {}

    def _get_state(self):
        """FIX: Implement proper State 2 with DC-SFC information"""
        curr_dc = self.dcs[self.current_dc_idx]
        
        # State 1: DC Info (unchanged)
        s1 = [curr_dc.cpu, curr_dc.storage]
        installed = [0] * config.NUM_VNF_TYPES
        idle = [0] * config.NUM_VNF_TYPES
        for v in curr_dc.installed_vnfs:
            v_idx = config.VNF_TYPES.index(v.vnf_type)
            installed[v_idx] += 1
            if v.is_idle():
                idle[v_idx] += 1
        s1.extend(installed)
        s1.extend(idle)
        
        # State 2: DC-SFC Info (FIXED!)
        # Matrix [|S| x (1 + 2*|V|)]
        # Mỗi hàng: [SFC_active_flag, allocated_vnfs..., remaining_vnfs...]
        s2 = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES))
        
        for i, s_type in enumerate(config.SFC_TYPES):
            # Lấy các request của type này có liên quan đến DC hiện tại
            related_reqs = [r for r in self.sfc_manager.active_requests 
                          if r.type == s_type and self._dc_involved_in_request(r, curr_dc.id)]
            
            if related_reqs:
                s2[i, 0] = 1  # Flag: DC này đang xử lý SFC type này
                
                # Thống kê VNF đã allocated và còn pending
                allocated_vnfs = np.zeros(config.NUM_VNF_TYPES)
                remaining_vnfs = np.zeros(config.NUM_VNF_TYPES)
                
                for req in related_reqs:
                    # VNFs đã được allocated
                    for vnf_name, dc_id in req.placed_vnfs:
                        if dc_id == curr_dc.id:
                            v_idx = config.VNF_TYPES.index(vnf_name)
                            allocated_vnfs[v_idx] += 1
                    
                    # VNFs còn phải allocated
                    next_vnf = req.get_next_vnf()
                    if next_vnf:
                        v_idx = config.VNF_TYPES.index(next_vnf)
                        remaining_vnfs[v_idx] += 1
                
                s2[i, 1:config.NUM_VNF_TYPES+1] = allocated_vnfs
                s2[i, config.NUM_VNF_TYPES+1:] = remaining_vnfs
        
        # State 3: Global Info (unchanged)
        state_3 = self.sfc_manager.get_global_state_info().astype(np.float32)
        
        return (np.array(s1, dtype=np.float32), s2.flatten().astype(np.float32), state_3)
    
    def _dc_involved_in_request(self, request, dc_id):
        """Check if DC is involved in processing this request"""
        # DC liên quan nếu: (1) là source/dest, (2) đã placed VNF, (3) có VNF idle phù hợp
        if request.source == dc_id or request.destination == dc_id:
            return True
        
        for _, placed_dc_id in request.placed_vnfs:
            if placed_dc_id == dc_id:
                return True
        
        return False

    def step(self, action):
        reward = 0
        curr_dc = self.dcs[self.current_dc_idx]

        pending_requests_count = sum(1 for req in self.sfc_manager.active_requests 
                                    if not req.is_completed and not req.is_dropped)
        
        # --- 1. Perform Action ---
        action_type = "WAIT"
        target_vnf_idx = -1
        
        if action == 0:
            reward = config.REWARD_WAIT if pending_requests_count > 0 else 0.0
        elif 1 <= action <= config.NUM_VNF_TYPES:
            action_type = "UNINSTALL"
            target_vnf_idx = action - 1
        else:
            action_type = "ALLOC"
            target_vnf_idx = action - (config.NUM_VNF_TYPES + 1)
        
        target_vnf_name = config.VNF_TYPES[target_vnf_idx] if target_vnf_idx >= 0 else None

        # UNINSTALL Logic
        if action_type == "UNINSTALL":
            removed = False
            for v in curr_dc.installed_vnfs[:]:  # Copy list to avoid modification during iteration
                if v.vnf_type == target_vnf_name and v.is_idle():
                    curr_dc.installed_vnfs.remove(v)
                    curr_dc.release_resources(target_vnf_name)
                    reward = config.REWARD_UNINSTALL_REQ 
                    removed = True
                    break
            if not removed: 
                reward = config.REWARD_INVALID

        # ALLOC Logic
        elif action_type == "ALLOC":
            if curr_dc.has_resources(target_vnf_name):
                candidates = []
                for req in self.sfc_manager.active_requests:
                    if req.is_completed or req.is_dropped:
                        continue
                    next_vnf = req.get_next_vnf()
                    if next_vnf == target_vnf_name:
                        p = calculate_priority(req, curr_dc.id)
                        candidates.append((p, req))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_req = candidates[0][1]
                    
                    # Find or create VNF instance
                    vnf_instance = None
                    for v in curr_dc.installed_vnfs:
                        if v.vnf_type == target_vnf_name and v.is_idle():
                            vnf_instance = v
                            break
                    
                    if not vnf_instance:
                        curr_dc.consume_resources(target_vnf_name)
                        vnf_instance = VNFInstance(target_vnf_name, curr_dc.id)
                        curr_dc.installed_vnfs.append(vnf_instance)
                    
                    vnf_instance.assign(best_req.id, config.VNF_SPECS[target_vnf_name]['proc_time'])
                    best_req.advance_chain(curr_dc.id)
                    
                    if best_req.is_completed:
                        reward = config.REWARD_SATISFIED
                        self.total_accepted_requests += 1
                    else:
                        reward = 0.2  # Small positive reward for progress
                else:
                    reward = config.REWARD_INVALID
            else:
                reward = config.REWARD_INVALID

        # --- 2. Update Time & Environment ---
        self.action_step_counter += 1
        
        if self.action_step_counter >= config.ACTIONS_PER_TIME_STEP:
            self.sim_time_ms += 1
            self.action_step_counter = 0
            
            # Tick resources
            for dc in self.dcs:
                for v in dc.installed_vnfs: 
                    v.tick()
            
            # Check drops
            dropped_this_step = 0
            for req in self.sfc_manager.active_requests:
                was_dropped = req.is_dropped
                req.update_time()
                if not was_dropped and req.is_dropped:
                    dropped_this_step += 1
            
            if dropped_this_step > 0:
                reward += config.REWARD_DROPPED * dropped_this_step
            
            self.sfc_manager.clean_requests()
            
            # Generate traffic
            if (self.sim_time_ms < config.TRAFFIC_STOP_TIME and 
                self.sim_time_ms % config.TRAFFIC_GEN_INTERVAL == 0):
                generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
                self.total_generated_requests += generated

        # --- 3. Next DC ---
        self.current_dc_idx = (self.current_dc_idx + 1) % self.num_active_dcs

        # --- 4. Done Condition ---
        done = (self.sim_time_ms >= config.MAX_SIM_TIME_PER_EPISODE) or \
               (self.sim_time_ms > config.TRAFFIC_STOP_TIME and 
                len(self.sfc_manager.active_requests) == 0)

        acc_ratio = 0
        if self.total_generated_requests > 0:
            acc_ratio = (self.total_accepted_requests / self.total_generated_requests) * 100

        return self._get_state(), reward, done, False, {"acc_ratio": acc_ratio}