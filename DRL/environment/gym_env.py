import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

import config
from spaces.dc import DataCenter
from spaces.vnf import VNFInstance
from spaces.sfc_manager import SFC_Manager
from environment.utils import calculate_priority, get_valid_actions_mask, get_dc_sfc_state_info

class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()
        
        self.action_space = spaces.Discrete(config.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=10000, shape=(2 * config.NUM_VNF_TYPES + 2,)),
            spaces.Box(low=0, high=10000, shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),)),
            spaces.Box(low=0, high=10000, shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),))
        ))
        
        self.sfc_manager = SFC_Manager()
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
        
        self.sfc_manager = SFC_Manager()
        self.sim_time_ms = 0
        self.action_step_counter = 0
        self.current_dc_idx = 0
        
        self.total_generated_requests = 0
        self.total_accepted_requests = 0
        
        generated = self.sfc_manager.generate_requests(self.sim_time_ms, self.num_active_dcs)
        self.total_generated_requests += generated
        
        return self._get_state(), {}

    def _get_valid_actions_mask(self):
        curr_dc = self.dcs[self.current_dc_idx]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)

    def _get_state(self):
        """
        Constructs the state observation.
        State 2 logic is delegated to utils.py
        """
        curr_dc = self.dcs[self.current_dc_idx]
        
        # State 1: DC Info (Local)
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
        
        # State 2: DC-SFC Info (Delegated to utils)
        s2_flattened = get_dc_sfc_state_info(curr_dc, self.sfc_manager.active_requests)
        
        # State 3: Global Info (From SFC Manager)
        state_3 = self.sfc_manager.get_global_state_info().astype(np.float32)
        
        return (np.array(s1, dtype=np.float32), s2_flattened, state_3)

    def step(self, action):
        reward = 0
        curr_dc = self.dcs[self.current_dc_idx]

        # Count active pending requests
        pending_requests_count = sum(1 for req in self.sfc_manager.active_requests 
                                    if not req.is_completed and not req.is_dropped)
        
        # --- 1. Decode and Perform Action ---
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

            # Check if uninstalling a needed VNF
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
            # Check demand
            has_demand = False
            for req in self.sfc_manager.active_requests:
                if not req.is_completed and not req.is_dropped:
                    if req.get_next_vnf() == target_vnf_name:
                        has_demand = True
                        break

            if not has_demand:
                reward = config.REWARD_WAIT
            elif not curr_dc.has_resources(target_vnf_name):
                reward = config.REWARD_INVALID
            else:
                # Select highest priority request
                candidates = []
                for req in self.sfc_manager.active_requests:
                    if not req.is_completed and not req.is_dropped:
                        if req.get_next_vnf() == target_vnf_name:
                            # Use calculate_priority from utils
                            prio = calculate_priority(req, curr_dc.id)
                            candidates.append((prio, req))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_req = candidates[0][1]

                    # Create or find VNF instance
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
        self.action_step_counter += 1
        
        if self.action_step_counter >= config.ACTIONS_PER_TIME_STEP:
            self.sim_time_ms += 1
            self.action_step_counter = 0
            
            # Tick VNFs
            for dc in self.dcs:
                for v in dc.installed_vnfs:
                    v.tick()
            
            # Update requests
            for req in self.sfc_manager.active_requests:
                was_dropped = req.is_dropped
                req.update_time()
                if not was_dropped and req.is_dropped:
                    reward += config.REWARD_DROPPED
            
            self.sfc_manager.clean_requests()
            
            # Generate traffic
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

        # Calculate metrics
        acc_ratio = 0
        if self.total_generated_requests > 0:
            acc_ratio = (self.total_accepted_requests / self.total_generated_requests) * 100

        next_state = self._get_state()
        action_mask = self._get_valid_actions_mask()
        
        return next_state, reward, done, False, {
            "acc_ratio": acc_ratio, 
            "action_mask": action_mask,
            "sim_time_ms": self.sim_time_ms,
            "action_in_timestep": self.action_step_counter
        }
    
    def _get_timing_info(self):
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