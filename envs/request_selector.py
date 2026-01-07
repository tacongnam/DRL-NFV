import config

class RequestSelector:
    @staticmethod
    def select_best(vnf_type, dc_id, active_requests):
        candidates = []
        for req in active_requests:
            if (not req.is_completed and not req.is_dropped and 
                req.get_next_vnf() == vnf_type):
                priority = RequestSelector._calculate_priority(req, dc_id)
                candidates.append((priority, req))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    @staticmethod
    def _calculate_priority(req, dc_id):
        p1 = req.elapsed_time - req.max_delay
        
        last_dc = req.get_last_placed_dc()
        p2 = 0.0
        if last_dc is not None:
            p2 = (config.PRIORITY_P2_SAME_DC if last_dc == dc_id 
                  else config.PRIORITY_P2_DIFF_DC)
        
        remaining = req.get_remaining_time()
        p3 = 0.0
        if remaining < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining + config.EPSILON_SMALL)
        
        return p1 + p2 + p3