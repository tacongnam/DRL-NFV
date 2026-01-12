import config

class ActionHandler:
    tracker = None
    
    @staticmethod
    def execute(env, dc, action):
        if action == 0:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('WAIT', config.REWARD_WAIT)
            return config.REWARD_WAIT, False
        
        vnf_type = (action - 1) % config.NUM_VNF_TYPES
        is_allocate = action > config.NUM_VNF_TYPES
        
        if is_allocate:
            return ActionHandler._handle_allocation(env, dc, vnf_type)
        else:
            return ActionHandler._handle_uninstall(env, dc, vnf_type)
    
    @staticmethod
    def _handle_allocation(env, dc, vnf_type):
        from envs.request_selector import RequestSelector
        
        req = RequestSelector.select_best(vnf_type, dc.id, env.sfc_manager.active_requests)
        if req is None:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_WAIT, 'no_request_needs_vnf')
            return config.REWARD_WAIT, False
        
        # Check if request is about to expire
        if req.get_remaining_time() < 1.0:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_WAIT, 'request_expiring')
            return config.REWARD_WAIT, False
        
        vnf = dc._get_idle_vnf(vnf_type)
        installed_new = False
        
        if vnf is None:
            if not dc.has_resources(vnf_type):
                if ActionHandler.tracker:
                    ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_INVALID, 'no_dc_resources')
                return config.REWARD_INVALID, False
            vnf = dc.install_vnf(vnf_type)
            if vnf is None:
                if ActionHandler.tracker:
                    ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_INVALID, 'install_failed')
                return config.REWARD_INVALID, False
            installed_new = True
        
        last_dc = req.get_last_placed_dc()
        prop_delay = 0.0
        path = []
        
        if last_dc != dc.id:
            success, path, delay = env.topology.allocate_bandwidth(
                last_dc, dc.id, req.bandwidth
            )
            
            if not success:
                if installed_new:
                    dc.uninstall_vnf(vnf_type)
                if ActionHandler.tracker:
                    ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_INVALID, 'no_bandwidth')
                return config.REWARD_INVALID, False
            
            prop_delay = delay
            req.allocated_paths.append((last_dc, dc.id, req.bandwidth, path))
        
        proc_time = vnf.get_processing_time(dc.delay if dc.delay else 0)
        
        # Check if allocation will exceed deadline
        potential_elapsed = req.elapsed_time + prop_delay + proc_time
        if potential_elapsed > req.max_delay:
            # Rollback
            if last_dc != dc.id and path:
                env.topology.release_bandwidth_on_path(path, req.bandwidth)
                req.allocated_paths.pop()
            if installed_new:
                dc.uninstall_vnf(vnf_type)
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_WAIT, 'would_exceed_deadline')
            return config.REWARD_WAIT, False
        
        dc_delay = dc.delay if dc.delay else 0.0
        vnf.assign(req.id, dc_delay, waiting_time=0.0)
        
        req.advance_chain(dc.id, prop_delay, proc_time, vnf)
        is_complete = req.check_completion()
        
        if is_complete:
            for _, _, bw, p in req.allocated_paths:
                env.topology.release_bandwidth_on_path(p, bw)
            req.allocated_paths.clear()
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_SATISFIED, 'sfc_completed')
            return config.REWARD_SATISFIED, True
        else:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('ALLOCATE', config.REWARD_WAIT, 'vnf_placed')
            return config.REWARD_WAIT, False
    
    @staticmethod
    def _handle_uninstall(env, dc, vnf_type):
        success = dc.uninstall_vnf(vnf_type)
        if success:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('UNINSTALL', config.REWARD_WAIT, 'uninstall_success')
            return config.REWARD_WAIT, False
        else:
            if ActionHandler.tracker:
                ActionHandler.tracker.track_action('UNINSTALL', config.REWARD_INVALID, 'no_idle_vnf')
            return config.REWARD_INVALID, False