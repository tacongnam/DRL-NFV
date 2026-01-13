import numpy as np
from abc import ABC, abstractmethod
import config

class DCSelector(ABC):
    @abstractmethod
    def get_dc_order(self, dcs, active_requests, topology):
        pass


class RandomSelector(DCSelector):
    def get_dc_order(self, dcs, active_requests, topology):
        dc_ids = [dc.id for dc in dcs]
        np.random.shuffle(dc_ids)
        return dc_ids


class PrioritySelector(DCSelector):
    """Paper-based DC priority: source DC + shortest path DCs"""
    def get_dc_order(self, dcs, active_requests, topology):
        if not active_requests:
            return [dc.id for dc in dcs]
        
        # Find request with minimum remaining E2E latency (most urgent)
        urgent_req = min(active_requests, key=lambda r: r.get_remaining_time())
        
        scores = {}
        for dc in dcs:
            score = 0.0
            
            # Highest priority: source DC of urgent request
            if dc.id == urgent_req.source:
                score += 1000.0
            
            # Medium priority: DCs on path from source to destination
            delay_to_src = topology.get_estimated_delay(urgent_req.source, dc.id)
            delay_to_dst = topology.get_estimated_delay(dc.id, urgent_req.destination)
            
            if delay_to_src != float('inf') and delay_to_dst != float('inf'):
                # Prefer DCs with lower total delay (on shortest path)
                total_delay = delay_to_src + delay_to_dst
                score += 500.0 / (total_delay + 1.0)
            else:
                # Unreachable DCs get lowest priority
                score -= 1000.0
            
            # Small bonus for resource availability
            if dc.is_server:
                cpu_ratio = dc.cpu / config.MAX_CPU if dc.cpu else 0
                score += cpu_ratio * 10.0
            
            scores[dc.id] = score
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class VAESelector(DCSelector):
    def __init__(self, vae_agent):
        self.agent = vae_agent
    
    def get_dc_order(self, dcs, active_requests, topology):
        from envs.observer import Observer
        
        if not active_requests:
            return [dc.id for dc in dcs]
        
        dc_states = Observer.get_all_dc_states(dcs, active_requests)
        values = self.agent.predict_dc_values(dc_states)
        ranked_indices = np.argsort(values.flatten())[::-1]
        
        return [dcs[i].id for i in ranked_indices]