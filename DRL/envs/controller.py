# environment/controller.py
from DRL import config
from DRL.core.vnf import VNFInstance

class ActionController:
    """Controller to handle actions from DRL model"""
    
    def __init__(self, sfc_manager, topology_manager):
        self.manager = sfc_manager
        self.topology = topology_manager
        self.accepted_count = 0
        self.dropped_count = 0

    def execute_action(self, action, curr_dc):
        """
        Execute action.
        
        Args:
            action: Action ID (0: Wait, 1-N: Uninstall, N+1-2N: Alloc)
            curr_dc: Current DataCenter
            
        Returns:
            (reward, is_sfc_completed)
        """
        # Action 0: WAIT
        if action == 0:
            return config.REWARD_WAIT, False
        
        vnf_type_idx = (action - 1) % config.NUM_VNF_TYPES
        is_alloc = action > config.NUM_VNF_TYPES
        
        if is_alloc:
            return self._handle_allocation(curr_dc, vnf_type_idx)
        else:
            return self._handle_uninstall(curr_dc, vnf_type_idx)

    def _handle_allocation(self, dc, vnf_type):
        """Handle Allocation action"""
        # 1. Find highest priority request that needs this VNF
        candidates = []
        for req in self.manager.active_requests:
            if not req.is_completed and not req.is_dropped and req.get_next_vnf() == vnf_type:
                priority = self._calculate_vnf_priority(req, dc.id)
                candidates.append((priority, req))
        
        if not candidates:
            # No request needs this VNF
            return config.REWARD_INVALID, False
        
        # Sort by priority descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_req = candidates[0][1]
        
        # 2. Find or create VNF instance
        vnf_instance = dc.get_idle_vnf(vnf_type)
        
        if vnf_instance is None:
            # Need to install new VNF
            if not dc.has_resources(vnf_type):
                return config.REWARD_INVALID, False
            
            # Consume resources and create new VNF
            dc.consume_resources(vnf_type)
            vnf_instance = VNFInstance(vnf_type, dc.id)
            dc.installed_vnfs.append(vnf_instance)
        
        # 3. Calculate delays
        prop_delay = 0.0
        last_dc = best_req.get_last_placed_dc()
        if last_dc is not None and last_dc != dc.id:
            prop_delay = self.topology.get_propagation_delay(last_dc, dc.id)
        
        # Processing time = startup_time + dc.delay (calculated in VNF.assign)
        # DC delay
        dc_delay = dc.delay if dc.delay is not None else 0.0
        waiting_time = 0.0
        
        # 4. Assign VNF with dc_delay (VNF will calculate startup_time + dc_delay)
        vnf_instance.assign(best_req.id, dc_delay, waiting_time)
        
        # 5. Advance chain (save delay info and VNF instance)
        # Get actual processing time from VNF instance
        total_proc_delay = vnf_instance.get_processing_time(dc_delay)
        best_req.advance_chain(dc.id, prop_delay, total_proc_delay, vnf_instance)
        
        # 6. Check completion (check after advance)
        best_req.check_completion()
        
        if best_req.is_completed:
            self.accepted_count += 1
            return config.REWARD_SATISFIED, True
        
        # Partial reward for progressing in chain
        return config.REWARD_SATISFIED / 2.0, False

    def _handle_uninstall(self, dc, vnf_type):
        """Handle Uninstall action"""
        # Check if this VNF is needed by any request
        is_needed = any(r.get_next_vnf() == vnf_type for r in self.manager.active_requests)
        
        if is_needed:
            # VNF is needed for future requests
            return config.REWARD_UNINSTALL_NEEDED, False
        
        # Check if there's an idle VNF to uninstall
        vnf_to_remove = None
        for vnf in dc.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                vnf_to_remove = vnf
                break
        
        if vnf_to_remove:
            dc.installed_vnfs.remove(vnf_to_remove)
            dc.release_resources(vnf_type)
            return 0.0, False  # Neutral reward
        
        # Cannot uninstall (VNF busy or doesn't exist)
        return config.REWARD_INVALID, False

    def _calculate_vnf_priority(self, req, dc_id):
        """
        Calculate VNF priority to select most suitable request.
        
        P = P1 + P2 + P3
        - P1: elapsed_time - max_delay (closer to deadline = higher priority)
        - P2: Affinity (is previous VNF in same DC?)
        - P3: Urgency (if remaining time < threshold)
        """
        # P1: Time priority
        p1 = req.elapsed_time - req.max_delay
        
        # P2: Affinity
        p2 = 0.0
        last_dc = req.get_last_placed_dc()
        if last_dc is not None:
            if last_dc == dc_id:
                p2 = config.PRIORITY_P2_SAME_DC
            else:
                p2 = config.PRIORITY_P2_DIFF_DC
        
        # P3: Urgency
        remaining_time = req.get_remaining_time()
        p3 = 0.0
        if remaining_time < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining_time + config.EPSILON_SMALL)
        
        return p1 + p2 + p3