import config

class ActionHandler:
    @staticmethod
    def execute(env, dc, action):
        if action == 0: return config.REWARD_WAIT, False
        
        vnf_type = (action - 1) % config.NUM_VNF_TYPES
        # Action > MAX_VNF -> Allocate, ngược lại -> Uninstall
        if action > config.NUM_VNF_TYPES:
            return ActionHandler._allocate(env, dc, vnf_type)
        
        # Uninstall
        return (config.REWARD_WAIT if dc.uninstall_vnf(vnf_type) else config.REWARD_INVALID), False

    @staticmethod
    def _select_best_request(vnf_type, dc_id, requests):
        """Tìm request có độ ưu tiên cao nhất."""
        best_req = None
        max_priority = -float('inf')

        for r in requests:
            if not r.is_completed and not r.is_dropped and r.get_next_vnf() == vnf_type:
                # P1: Time pressure
                p = -r.get_remaining_time() # Thời gian càng ít, priority càng cao (số âm lớn hơn)
                
                # P2: Locality (Thưởng lớn nếu cùng DC để tránh tốn băng thông)
                last_node = r.get_last_placed_dc()
                if last_node == dc_id:
                    p += 100.0 # Ưu tiên cực cao xử lý tại chỗ
                
                if p > max_priority:
                    max_priority = p
                    best_req = r
        return best_req

    @staticmethod
    def _allocate(env, dc, vnf_type):
        req = ActionHandler._select_best_request(vnf_type, dc.id, env.sfc_manager.active_requests)
        if not req: return config.REWARD_WAIT, False # Không có request nào cần

        # 1. Prepare VNF
        vnf = dc._get_idle_vnf(vnf_type)
        new_install = False
        if not vnf:
            if not dc.has_resources(vnf_type): return config.REWARD_INVALID, False
            vnf = dc.install_vnf(vnf_type)
            if not vnf: return config.REWARD_INVALID, False
            new_install = True

        # 2. Network Routing (Quan trọng cho Easy Mode)
        last_dc = req.get_last_placed_dc()
        path, prop_delay = [], 0.0
        
        if last_dc != dc.id:
            ok, path, prop_delay = env.topology.allocate_bandwidth(last_dc, dc.id, req.bandwidth)
            if not ok:
                # Rollback VNF nếu vừa cài mới
                if new_install: dc.uninstall_vnf(vnf_type)
                # Phạt RẤT NẶNG nếu Agent cố tình chọn đường cụt
                return config.REWARD_INVALID * 2.0, False

        # 3. Delay Check
        dc_delay = dc.delay or 0.0
        proc_time = vnf.get_processing_time(dc_delay)
        
        if req.elapsed_time + prop_delay + proc_time > req.max_delay:
            # Rollback Network & VNF
            if path: env.topology.release_bandwidth_on_path(path, req.bandwidth)
            if new_install: dc.uninstall_vnf(vnf_type)
            return config.REWARD_INVALID, False

        # 4. Commit
        if path:
            req.allocated_paths.append((last_dc, dc.id, req.bandwidth, path))
            
        vnf.assign(req.id, dc_delay)
        req.advance_chain(dc.id, prop_delay, proc_time, vnf)

        # Tính reward
        # Cost phạt nhẹ để khuyến khích tiết kiệm tài nguyên
        cost_penalty = 0.01 
        
        if req.check_completion():
            # Release bandwidth ngay khi xong để request khác dùng (quan trọng cho mạng nghẽn)
            for _, _, bw, p in req.allocated_paths:
                env.topology.release_bandwidth_on_path(p, bw)
            req.allocated_paths.clear()
            
            return config.REWARD_SATISFIED, True
        
        return config.REWARD_STEP_COMPLETED - cost_penalty, False