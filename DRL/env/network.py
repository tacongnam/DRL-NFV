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

    def _get_valid_actions_mask(self):
        """
        CRITICAL: Generate action mask to prevent invalid actions
        Returns boolean array where True = valid action
        """
        curr_dc = self.dcs[self.current_dc_idx]
        mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
        
        # Action 0: WAIT - always valid
        mask[0] = True
        
        # Actions 1 to NUM_VNF_TYPES: UNINSTALL
        for i in range(config.NUM_VNF_TYPES):
            vnf_type = config.VNF_TYPES[i]
            
            # Check if we have idle VNF of this type to uninstall
            has_idle = False
            for v in curr_dc.installed_vnfs:
                if v.vnf_type == vnf_type and v.is_idle():
                    has_idle = True
                    break
            
            # Only allow uninstall if: (1) has idle instance AND (2) not critically needed
            if has_idle:
                # Check if any active request urgently needs this VNF
                is_urgent = False
                for req in self.sfc_manager.active_requests:
                    if not req.is_completed and not req.is_dropped:
                        next_vnf = req.get_next_vnf()
                        if next_vnf == vnf_type:
                            # Check urgency: remaining time < threshold
                            remaining = req.max_delay - req.elapsed_time
                            if remaining < config.URGENCY_THRESHOLD:
                                is_urgent = True
                                break
                
                if not is_urgent:
                    mask[i + 1] = True
        
        # Actions (NUM_VNF_TYPES+1) to end: ALLOC
        for i in range(config.NUM_VNF_TYPES):
            vnf_type = config.VNF_TYPES[i]
            action_idx = config.NUM_VNF_TYPES + 1 + i
            
            # Check if: (1) has resources AND (2) some request needs this VNF
            if curr_dc.has_resources(vnf_type):
                has_demand = False
                for req in self.sfc_manager.active_requests:
                    if not req.is_completed and not req.is_dropped:
                        next_vnf = req.get_next_vnf()
                        if next_vnf == vnf_type:
                            has_demand = True
                            break
                
                if has_demand:
                    mask[action_idx] = True
        
        # Fallback: If no action is valid (shouldn't happen), allow WAIT
        if not np.any(mask):
            mask[0] = True
        
        return mask

    def _get_state(self):
        """
        CRITICAL FIX: Implement State 2 properly as per paper Section III.A
        State 2: DC-SFC Info - shows SFC processing stages at current DC
        """
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
        
        # State 2: DC-SFC Info (FIXED IMPLEMENTATION)
        # Matrix [|S| x (1 + 2*|V|)]
        # Each row: [SFC_type_indicator, already_allocated_VNFs..., remaining_VNFs...]
        s2 = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES))
        
        for i, s_type in enumerate(config.SFC_TYPES):
            # Get requests of this type that involve current DC
            relevant_reqs = []
            for req in self.sfc_manager.active_requests:
                if req.type == s_type and not req.is_completed and not req.is_dropped:
                    # Check if this DC is involved in the request
                    dc_involved = False
                    
                    # Check if any VNF already placed at this DC
                    for vnf_name, dc_id in req.placed_vnfs:
                        if dc_id == curr_dc.id:
                            dc_involved = True
                            break
                    
                    # Check if next VNF could be placed here
                    next_vnf = req.get_next_vnf()
                    if next_vnf and curr_dc.has_resources(next_vnf):
                        dc_involved = True
                    
                    if dc_involved:
                        relevant_reqs.append(req)
            
            if relevant_reqs:
                # Set SFC type indicator
                s2[i, 0] = len(relevant_reqs)  # Number of this SFC type being processed
                
                # Count allocated and remaining VNFs
                allocated_counts = np.zeros(config.NUM_VNF_TYPES)
                remaining_counts = np.zeros(config.NUM_VNF_TYPES)
                
                for req in relevant_reqs:
                    # Already allocated VNFs at this DC
                    for vnf_name, dc_id in req.placed_vnfs:
                        if dc_id == curr_dc.id:
                            v_idx = config.VNF_TYPES.index(vnf_name)
                            allocated_counts[v_idx] += 1
                    
                    # Remaining VNFs in the chain
                    for j in range(req.current_vnf_index, len(req.chain)):
                        vnf_name = req.chain[j]
                        v_idx = config.VNF_TYPES.index(vnf_name)
                        remaining_counts[v_idx] += 1
                
                s2[i, 1:config.NUM_VNF_TYPES+1] = allocated_counts
                s2[i, config.NUM_VNF_TYPES+1:] = remaining_counts
        
        # State 3: Global Info (unchanged)
        state_3 = self.sfc_manager.get_global_state_info().astype(np.float32)
        
        return (np.array(s1, dtype=np.float32), s2.flatten().astype(np.float32), state_3)

    def step(self, action):
        reward = 0
        curr_dc = self.dcs[self.current_dc_idx]

        # Count active pending requests
        pending_requests_count = sum(1 for req in self.sfc_manager.active_requests 
                                    if not req.is_completed and not req.is_dropped)
        
        # --- 1. Decode and Perform Action ---
         # --- 1. Decode and Perform Action (USING config rewards) ---
        reward = 0
        curr_dc = self.dcs[self.current_dc_idx]

        pending_requests_count = sum(
            1 for req in self.sfc_manager.active_requests
            if not req.is_completed and not req.is_dropped
        )

        action_type = "WAIT"
        target_vnf_idx = -1

        # WAIT ACTION
        if action == 0:
            if pending_requests_count == 0:
                reward = 0.0
            else:
                reward = config.REWARD_WAIT

        # UNINSTALL
        elif 1 <= action <= config.NUM_VNF_TYPES:
            action_type = "UNINSTALL"
            target_vnf_idx = action - 1

        # ALLOC
        else:
            action_type = "ALLOC"
            target_vnf_idx = action - (config.NUM_VNF_TYPES + 1)

        target_vnf_name = (
            config.VNF_TYPES[target_vnf_idx] if target_vnf_idx >= 0 else None
        )

        # --- UNINSTALL LOGIC ---
        if action_type == "UNINSTALL":

            removed = False
            needed = False

            # Kiểm tra uninstall VNF đang cần
            for req in self.sfc_manager.active_requests:
                if not req.is_completed and not req.is_dropped:
                    if req.get_next_vnf() == target_vnf_name:
                        needed = True
                        break

            if needed:
                reward = config.REWARD_UNINSTALL_REQ

            else:
                for v in curr_dc.installed_vnfs[:]:
                    if v.vnf_type == target_vnf_name and v.is_idle():
                        curr_dc.installed_vnfs.remove(v)
                        curr_dc.release_resources(target_vnf_name)
                        removed = True
                        reward = 0.0
                        break

                if not removed:
                    reward = 0.0  # exploration

        # --- ALLOC LOGIC ---
        elif action_type == "ALLOC":

            # Kiểm tra nhu cầu
            has_demand = False
            for req in self.sfc_manager.active_requests:
                if not req.is_completed and not req.is_dropped:
                    if req.get_next_vnf() == target_vnf_name:
                        has_demand = True
                        break

            if not has_demand:
                reward = config.REWARD_WAIT   # nhẹ, không phạt mạnh

            elif not curr_dc.has_resources(target_vnf_name):
                reward = config.REWARD_INVALID

            else:
                # Chọn request ưu tiên cao nhất
                candidates = []
                for req in self.sfc_manager.active_requests:
                    if not req.is_completed and not req.is_dropped:
                        if req.get_next_vnf() == target_vnf_name:
                            prio = calculate_priority(req, curr_dc.id)
                            candidates.append((prio, req))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_req = candidates[0][1]

                    # Tạo hoặc tìm VNF instance
                    vnf_instance = None
                    for v in curr_dc.installed_vnfs:
                        if v.vnf_type == target_vnf_name and v.is_idle():
                            vnf_instance = v
                            break

                    if not vnf_instance:
                        curr_dc.consume_resources(target_vnf_name)
                        vnf_instance = VNFInstance(target_vnf_name, curr_dc.id)
                        curr_dc.installed_vnfs.append(vnf_instance)

                    # Assign
                    proc_time = config.VNF_SPECS[target_vnf_name]['proc_time']
                    vnf_instance.assign(best_req.id, proc_time)
                    best_req.advance_chain(curr_dc.id)

                    if best_req.is_completed:
                        reward = config.REWARD_SATISFIED
                        self.total_accepted_requests += 1
                    else:
                        reward = config.REWARD_PARTIAL

        # --- 2. Update Environment Time Logic ---
        # Paper Section III.A.4: "The model performs A actions during each step, 
        # which is of duration T=1ms. Every action inference time stamp is 0.01ms."
        # 
        # This means:
        # - 1 timestep (T) = 1 ms
        # - A actions per timestep (A = 100 in paper, config.ACTIONS_PER_TIME_STEP in code)
        # - Each action takes 1ms / A = 0.01ms (inference time)
        # - Physical time only advances after A actions complete
        #
        # Current implementation:
        self.action_step_counter += 1  # Count actions in current timestep
        
        # Only advance simulation time after A actions
        if self.action_step_counter >= config.ACTIONS_PER_TIME_STEP:
            # === TIMESTEP COMPLETED: Advance physical time by 1ms ===
            self.sim_time_ms += 1
            self.action_step_counter = 0
            
            # Tick all VNF instances (decrement processing time)
            # Note: Processing times are in ms, so we decrement by T=1
            for dc in self.dcs:
                for v in dc.installed_vnfs:
                    v.tick()
            
            # Update all active requests' elapsed time
            # Check if any requests exceeded their E2E delay constraint
            for req in self.sfc_manager.active_requests:
                was_dropped = req.is_dropped
                req.update_time()  # Increments elapsed_time by T=1ms
                if not was_dropped and req.is_dropped:
                    reward += config.REWARD_DROPPED
            
            # Clean completed/dropped requests from active list
            self.sfc_manager.clean_requests()
            
            # Generate new traffic at specified intervals
            # Paper: "at the onset and intervals of every N=4 Steps"
            if (self.sim_time_ms < config.TRAFFIC_STOP_TIME and 
                self.sim_time_ms % config.TRAFFIC_GEN_INTERVAL == 0):
                generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
                self.total_generated_requests += generated

        # --- 3. Move to next DC ---
        self.current_dc_idx = (self.current_dc_idx + 1) % self.num_active_dcs

        # --- 4. Determine if episode is done ---
        done = (self.sim_time_ms >= config.MAX_SIM_TIME_PER_EPISODE) or \
               (self.sim_time_ms > config.TRAFFIC_STOP_TIME and 
                len(self.sfc_manager.active_requests) == 0)

        # Calculate current acceptance ratio
        acc_ratio = 0
        if self.total_generated_requests > 0:
            acc_ratio = (self.total_accepted_requests / self.total_generated_requests) * 100

        # Return state and action mask
        next_state = self._get_state()
        action_mask = self._get_valid_actions_mask()
        
        return next_state, reward, done, False, {
            "acc_ratio": acc_ratio, 
            "action_mask": action_mask,
            "sim_time_ms": self.sim_time_ms,
            "action_in_timestep": self.action_step_counter
        }
    
    def get_timing_info(self):
        """Get detailed timing information for debugging"""
        return {
            "sim_time_ms": self.sim_time_ms,
            "action_counter": self.action_step_counter,
            "actions_per_timestep": config.ACTIONS_PER_TIME_STEP,
            "action_inference_time_ms": config.TIME_STEP / config.ACTIONS_PER_TIME_STEP,
            "next_time_advance_in": config.ACTIONS_PER_TIME_STEP - self.action_step_counter,
            "traffic_gen_interval": config.TRAFFIC_GEN_INTERVAL,
            "next_traffic_gen_at": ((self.sim_time_ms // config.TRAFFIC_GEN_INTERVAL) + 1) * config.TRAFFIC_GEN_INTERVAL
        }