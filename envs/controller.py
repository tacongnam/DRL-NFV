import config
from core.vnf import VNFInstance

class ActionController:
    def __init__(self, sfc_manager, topology_manager):
        self.manager = sfc_manager
        self.topology = topology_manager
        self.accepted_count = 0
        self.dropped_count = 0

    def execute_action(self, action, curr_dc):
        if action == 0:
            return config.REWARD_WAIT, False
        
        vnf_type_idx = (action - 1) % config.NUM_VNF_TYPES
        is_alloc = action > config.NUM_VNF_TYPES
        
        if is_alloc:
            return self._handle_allocation(curr_dc, vnf_type_idx)
        else:
            return self._handle_uninstall(curr_dc, vnf_type_idx)

    def _handle_allocation(self, dc, vnf_type):
        candidates = []
        for req in self.manager.active_requests:
            if not req.is_completed and not req.is_dropped and req.get_next_vnf() == vnf_type:
                priority = self._calculate_priority(req, dc.id)
                candidates.append((priority, req))
        
        if not candidates:
            return config.REWARD_INVALID, False
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_req = candidates[0][1]
        
        vnf_instance = dc.get_idle_vnf(vnf_type)
        
        if vnf_instance is None:
            if not dc.has_resources(vnf_type):
                return config.REWARD_INVALID, False
            
            dc.consume_resources(vnf_type)
            vnf_instance = VNFInstance(vnf_type, dc.id)
            dc.installed_vnfs.append(vnf_instance)
        
        last_dc = best_req.get_last_placed_dc()
        prop_delay = 0.0
        hop_count = 0
        
        if last_dc != dc.id:
            success, path, delay = self.topology.consume_bandwidth(
                last_dc, dc.id, best_req.bandwidth
            )
            
            if not success:
                if vnf_instance in dc.installed_vnfs and vnf_instance.is_idle():
                    dc.installed_vnfs.remove(vnf_instance)
                    dc.release_resources(vnf_type)
                return config.REWARD_INVALID, False
            
            prop_delay = delay
            hop_count = len(path) - 1
            
            # Track for cleanup
            best_req.allocated_paths.append((last_dc, dc.id, best_req.bandwidth))
        
        dc_delay = dc.delay if dc.delay is not None else 0.0
        vnf_instance.assign(best_req.id, dc_delay, waiting_time=0.0)
        
        total_proc_delay = vnf_instance.get_processing_time(dc_delay)
        best_req.advance_chain(dc.id, prop_delay, total_proc_delay, vnf_instance)
        
        best_req.check_completion()
        
        base_reward = config.REWARD_SATISFIED if best_req.is_completed else config.REWARD_SATISFIED / 2.0
        delay_penalty = config.ALPHA_DELAY_PENALTY * prop_delay
        hop_penalty = config.BETA_HOP_PENALTY * hop_count
        
        reward = base_reward - delay_penalty - hop_penalty
        
        if best_req.is_completed:
            self.accepted_count += 1
            self._release_request_bandwidth(best_req)
        
        return reward, best_req.is_completed

    def _handle_uninstall(self, dc, vnf_type):
        is_needed = any(r.get_next_vnf() == vnf_type for r in self.manager.active_requests)
        
        if is_needed:
            return config.REWARD_UNINSTALL_NEEDED, False
        
        vnf_to_remove = None
        for vnf in dc.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                vnf_to_remove = vnf
                break
        
        if vnf_to_remove:
            dc.installed_vnfs.remove(vnf_to_remove)
            dc.release_resources(vnf_type)
            return 0.0, False
        
        return config.REWARD_INVALID, False

    def _release_request_bandwidth(self, req):
        for u, v, bw in req.allocated_paths:
            self.topology.release_bandwidth(u, v, bw)
        req.allocated_paths.clear()

    def _calculate_priority(self, req, dc_id):
        p1 = req.elapsed_time - req.max_delay
        
        p2 = 0.0
        last_dc = req.get_last_placed_dc()
        if last_dc is not None:
            p2 = config.PRIORITY_P2_SAME_DC if last_dc == dc_id else config.PRIORITY_P2_DIFF_DC
        
        remaining_time = req.get_remaining_time()
        p3 = 0.0
        if remaining_time < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining_time + config.EPSILON_SMALL)
        
        return p1 + p2 + p3