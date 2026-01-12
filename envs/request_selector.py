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
        # P1: Paper formula - time pressure (elapsed - max_delay)
        # Higher = more urgent (negative values mean still have time)
        p1 = req.elapsed_time - req.max_delay
        
        # P2: Paper formula - locality bonus/penalty
        last_dc = req.get_last_placed_dc()
        p2 = 0.0
        if last_dc is not None:
            if last_dc == dc_id:
                # Same DC: huge bonus (paper uses positive value)
                p2 = config.PRIORITY_P2_SAME_DC  # 10.0
            else:
                # Different DC: penalty (paper uses negative value)
                p2 = config.PRIORITY_P2_DIFF_DC  # -10.0
        
        # P3: Paper formula - urgency boost for critical requests
        remaining = req.get_remaining_time()
        p3 = 0.0
        if remaining < config.URGENCY_THRESHOLD:
            # Paper uses: C / (remaining + epsilon)
            p3 = config.URGENCY_CONSTANT_C / (remaining + config.EPSILON_SMALL)
        
        return p1 + p2 + p3