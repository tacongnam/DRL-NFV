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
        
        # Action Space & Observation Space (Giữ nguyên)
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
        
        # State tracking
        self.current_dc_idx = 0
        self.sim_time_ms = 0        # Thời gian mô phỏng (ms)
        self.action_step_counter = 0 # Đếm số action trong 1 ms (max 100)
        
        # Metrics for AR
        self.total_generated_requests = 0
        self.total_accepted_requests = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 1. Dynamic Environment: Randomize DC count per episode
        self.num_active_dcs = np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(self.num_active_dcs)]
        self.topology = nx.complete_graph(self.num_active_dcs)
        
        # 2. Reset SFC Manager & Time
        self.sfc_manager = SFCManager()
        self.sim_time_ms = 0
        self.action_step_counter = 0
        self.current_dc_idx = 0
        
        # 3. Reset Metrics
        self.total_generated_requests = 0
        self.total_accepted_requests = 0
        
        # 4. Initial Traffic Generation (At onset)
        generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
        self.total_generated_requests += generated
        
        return self._get_state(), {}

    def _get_state(self):
        # Đảm bảo index nằm trong phạm vi DC hiện có
        curr_dc = self.dcs[self.current_dc_idx]
        
        # State 1: DC Info
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
        
        # State 2: DC-SFC Info (Simplified placeholder)
        s2 = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES))
        
        # State 3: Global Info
        state_3 = self.sfc_manager.get_global_state_info().astype(np.float32)
        
        return (np.array(s1, dtype=np.float32), s2.flatten().astype(np.float32), state_3)

    def step(self, action):
        reward = 0
        curr_dc = self.dcs[self.current_dc_idx]

        pending_requests_count = 0
        for req in self.sfc_manager.active_requests:
            if not req.is_completed and not req.is_dropped:
                pending_requests_count += 1
        
        # --- 1. Perform Action ---
        action_type = "WAIT"
        target_vnf_idx = -1
        
        if action == 0:
            if pending_requests_count == 0:
                reward = 0.0
            else:
                reward = config.REWARD_WAIT
        elif 1 <= action <= config.NUM_VNF_TYPES:
            action_type = "UNINSTALL"
            target_vnf_idx = action - 1
        else:
            action_type = "ALLOC"
            target_vnf_idx = action - (config.NUM_VNF_TYPES + 1)
        
        target_vnf_name = config.VNF_TYPES[target_vnf_idx] if target_vnf_idx >= 0 else None

        # Logic xử lý hành động (giống phiên bản trước)
        if action_type == "UNINSTALL":
            removed = False
            for v in curr_dc.installed_vnfs:
                if v.vnf_type == target_vnf_name and v.is_idle():
                    curr_dc.installed_vnfs.remove(v)
                    curr_dc.release_resources(target_vnf_name)
                    reward = config.REWARD_UNINSTALL_REQ 
                    removed = True
                    break
            if not removed: reward = config.REWARD_INVALID

        elif action_type == "ALLOC":
            if curr_dc.has_resources(target_vnf_name):
                # ALGORITHM 1: Priority Selection
                candidates = []
                for req in self.sfc_manager.active_requests:
                    next_vnf = req.get_next_vnf()
                    if next_vnf == target_vnf_name:
                        p = calculate_priority(req, curr_dc.id)
                        candidates.append((p, req))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_req = candidates[0][1]
                    
                    vnf_instance = None
                    for v in curr_dc.installed_vnfs:
                        if v.vnf_type == target_vnf_name and v.is_idle():
                            vnf_instance = v
                            break
                    if not vnf_instance:
                        curr_dc.consume_resources(target_vnf_name)
                        from env.utils import VNFInstance
                        vnf_instance = VNFInstance(target_vnf_name, curr_dc.id)
                        curr_dc.installed_vnfs.append(vnf_instance)
                    
                    vnf_instance.assign(best_req.id, config.VNF_SPECS[target_vnf_name]['proc_time'])
                    best_req.advance_chain(curr_dc.id)
                    
                    if best_req.is_completed:
                        reward = config.REWARD_SATISFIED
                        self.total_accepted_requests += 1 # Update Metric
                    else:
                        reward = 0.5 
                else:
                    reward = config.REWARD_INVALID
            else:
                reward = config.REWARD_INVALID

        # --- 2. Update Time & Environment Logic (Paper: A=100 actions per 1ms) ---
        self.action_step_counter += 1
        
        # Chỉ cập nhật thời gian hệ thống sau mỗi 100 actions
        if self.action_step_counter >= config.ACTIONS_PER_TIME_STEP:
            self.sim_time_ms += 1
            self.action_step_counter = 0
            
            # a. Tick Resources & Requests
            for dc in self.dcs:
                for v in dc.installed_vnfs: v.tick()
            
            # b. Check Drops
            for req in self.sfc_manager.active_requests:
                req.update_time()
                if req.is_dropped:
                    reward += config.REWARD_DROPPED
            self.sfc_manager.clean_requests()
            
            # c. Generate Traffic (Paper: Intervals of every N=4 Steps)
            if (self.sim_time_ms < config.TRAFFIC_STOP_TIME and 
                self.sim_time_ms % config.TRAFFIC_GEN_INTERVAL == 0):
                generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
                self.total_generated_requests += generated

        # --- 3. Determine DC for next action ---
        # Paper says: "DCs' iteration order is defined by their priority points"
        # For simplicity in simulation: Round Robin or Logic based on requests
        self.current_dc_idx = (self.current_dc_idx + 1) % self.num_active_dcs

        # --- 4. Done Condition ---
        # Paper: "ends once there are no pending SFC requests"
        # Logic: Time > StopTime AND No active requests left
        done = (self.sim_time_ms >= config.MAX_SIM_TIME_PER_EPISODE) or \
               (self.sim_time_ms > config.TRAFFIC_STOP_TIME and len(self.sfc_manager.active_requests) == 0)

        # Tính Acceptance Ratio hiện tại để trả về info
        acc_ratio = 0
        if self.total_generated_requests > 0:
            acc_ratio = (self.total_accepted_requests / self.total_generated_requests) * 100

        return self._get_state(), reward, done, False, {"acc_ratio": acc_ratio}