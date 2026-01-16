import config

class ActionHandler:
    @staticmethod
    def execute(env, dc, action):
        if action == 0: return config.REWARD_WAIT, False
        
        vnf_type = (action - 1) % config.NUM_VNF_TYPES
        # Action > MAX_VNF -> Allocate, ngược lại -> Uninstall
        if action > config.NUM_VNF_TYPES:
            return ActionHandler._allocate(env, dc, vnf_type)
        
        # Uninstall nhanh gọn
        return (config.REWARD_WAIT if dc.uninstall_vnf(vnf_type) else config.REWARD_INVALID), False

    @staticmethod
    def _select_best_request(vnf_type, dc_id, requests):
        """Tìm request có độ ưu tiên cao nhất (P1+P2+P3) trong O(N)."""
        best_req = None
        max_priority = -float('inf')

        for r in requests:
            # Chỉ xét request chưa xong, chưa drop và đang cần vnf_type này
            if not r.is_completed and not r.is_dropped and r.get_next_vnf() == vnf_type:
                # P1: Time pressure
                p = r.elapsed_time - r.max_delay
                
                # P2: Locality (Thưởng nếu cùng DC, phạt nếu khác)
                last_node = r.get_last_placed_dc()
                p += config.PRIORITY_P2_SAME_DC if last_node == dc_id else config.PRIORITY_P2_DIFF_DC
                
                # P3: Urgency (Nếu sắp hết hạn thì ưu tiên cực cao)
                rem = r.get_remaining_time()
                if rem < config.URGENCY_THRESHOLD:
                    p += config.URGENCY_CONSTANT_C / (rem + config.EPSILON_SMALL)
                
                if p > max_priority:
                    max_priority = p
                    best_req = r
        return best_req

    @staticmethod
    def _allocate(env, dc, vnf_type):
        req = ActionHandler._select_best_request(vnf_type, dc.id, env.sfc_manager.active_requests)
        if not req: return config.REWARD_WAIT, False

        # 1. Prepare VNF (Reuse or Install)
        vnf = dc._get_idle_vnf(vnf_type)
        new_install = False
        if not vnf:
            if not dc.has_resources(vnf_type): return config.REWARD_INVALID, False
            vnf = dc.install_vnf(vnf_type)
            if not vnf: return config.REWARD_INVALID, False
            new_install = True

        # 2. Calculate Cost (Resource + Startup) & Penalty
        specs = config.VNF_SPECS[vnf_type]
        cost = (specs['cpu'] * dc.cost_c) + (specs['ram'] * dc.cost_r) + (specs['storage'] * dc.cost_h)
        if new_install and len(dc.installed_vnfs) == 1:
            cost += dc.initial_resources['cpu'] * dc.cost_c * 0.1
        penalty = cost * config.ALPHA_COST_PENALTY

        # 3. Network Routing & Deadline Check
        last_dc = req.get_last_placed_dc()
        path, prop_delay = [], 0.0
        
        if last_dc != dc.id:
            ok, path, prop_delay = env.topology.allocate_bandwidth(last_dc, dc.id, req.bandwidth)
            if not ok:
                if new_install: dc.uninstall_vnf(vnf_type)
                return config.REWARD_INVALID, False
            req.allocated_paths.append((last_dc, dc.id, req.bandwidth, path))

        # Check if total delay exceeds deadline
        dc_delay = dc.delay or 0.0
        proc_time = vnf.get_processing_time(dc_delay)
        
        if req.elapsed_time + prop_delay + proc_time > req.max_delay:
            # Rollback
            if path: 
                env.topology.release_bandwidth_on_path(path, req.bandwidth)
                req.allocated_paths.pop()
            if new_install: dc.uninstall_vnf(vnf_type)
            return config.REWARD_INVALID, False

        # 4. Commit & Reward
        vnf.assign(req.id, dc_delay)
        req.advance_chain(dc.id, prop_delay, proc_time, vnf)

        if req.check_completion():
            # Release bandwidth immediately upon completion
            for _, _, bw, p in req.allocated_paths:
                env.topology.release_bandwidth_on_path(p, bw)
            req.allocated_paths.clear()
            
            bonus = (req.max_delay - req.elapsed_time) / req.max_delay * 10.0
            return config.REWARD_SATISFIED + bonus - penalty, True
        
        return config.REWARD_STEP_COMPLETED - penalty, False