import numpy as np
import config

class ActionController:
    def __init__(self, sfc_manager):
        self.manager = sfc_manager
        self.accepted_count = 0

    def execute_action(self, action, curr_dc):
        # Action Map: 0:Wait, 1..N:Uninstall, N+1..2N:Alloc
        if action == 0:
            return config.REWARD_WAIT, False

        vnf_idx = (action - 1) % config.NUM_VNF_TYPES
        vnf_name = config.VNF_TYPES[vnf_idx]
        is_alloc = action > config.NUM_VNF_TYPES

        if is_alloc:
            return self._handle_allocation(curr_dc, vnf_name)
        else:
            return self._handle_uninstall(curr_dc, vnf_name)

    def _handle_allocation(self, dc, vnf_name):
        # 1. Find Best Request based on Priority P
        candidates = []
        for r in self.manager.active_requests:
            if r.get_next_vnf() == vnf_name:
                p = self._calculate_priority(r, dc.id)
                candidates.append((p, r))
        
        if not candidates:
            return config.REWARD_WAIT, False # Resource waste if alloc but no demand

        # Sort by Priority Descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_req = candidates[0][1]

        # 2. Check/Consume Resources
        if not dc.consume_resources(vnf_name):
             # Try to find idle VNF
            found_idle = False
            for v in dc.installed_vnfs:
                if v.vnf_type == vnf_name and v.is_idle():
                    found_idle = True
                    break
            if not found_idle:
                return config.REWARD_INVALID, False
        else:
            # Install new VNF instance
            from spaces.vnf import VNFInstance
            new_v = VNFInstance(vnf_name, dc.id)
            dc.installed_vnfs.append(new_v)

        # 3. Assign
        # Find the VNF instance (either newly created or idle)
        target_v = None
        for v in dc.installed_vnfs:
            if v.vnf_type == vnf_name and v.is_idle():
                target_v = v
                break
        
        if target_v:
            proc_time = config.VNF_SPECS[vnf_name]['proc_time']
            # Chuyển đổi proc_time sang số tick (làm tròn lên)
            ticks = int(np.ceil(proc_time / (config.TIME_STEP * 0.001))) # Assuming time_step in ms vs sec? 
            # Note: Paper says 1ms step. Proc time e.g. 0.06ms? 
            # If proc_time is ms in config:
            ticks = max(1, int(proc_time)) # Simple mapping
            
            target_v.assign(best_req.id, ticks)
            best_req.advance_chain(dc.id)
            
            if best_req.is_completed:
                self.accepted_count += 1
                return config.REWARD_SATISFIED, True
            return config.REWARD_SATISFIED / 2.0, False # Partial reward
            
        return config.REWARD_INVALID, False

    def _handle_uninstall(self, dc, vnf_name):
        # Check if needed by pending requests (Paper logic)
        is_needed = any(r.get_next_vnf() == vnf_name for r in self.manager.active_requests)
        if is_needed:
            return config.REWARD_UNINSTALL_REQ, False

        # Try remove idle
        for v in dc.installed_vnfs:
            if v.vnf_type == vnf_name and v.is_idle():
                dc.installed_vnfs.remove(v)
                dc.release_resources(vnf_name)
                return 0.0, False # Neutral or small positive for saving? Paper says 0 or penalty if wrong.
        
        return config.REWARD_INVALID, False # Cannot uninstall non-existent/busy

    def _calculate_priority(self, req, dc_id):
        # P1: Time
        p1 = req.elapsed_time - req.max_delay
        # P2: Affinity
        p2 = 0
        if req.placed_vnfs:
            if req.placed_vnfs[-1][1] == dc_id:
                p2 = config.PRIORITY_P2_SAME_DC
            else:
                p2 = config.PRIORITY_P2_DIFF_DC
        # P3: Urgency
        rem = req.max_delay - req.elapsed_time
        if rem < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (rem + config.EPSILON)
        else:
            p3 = 0
        return p1 + p2 + p3