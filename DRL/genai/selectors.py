import numpy as np
from abc import ABC, abstractmethod

class DCSelector(ABC):
    """Base class for DC selection strategies"""
    
    @abstractmethod
    def get_dc_order(self, dcs, active_requests, topology):
        """
        Return ordered list of DC IDs to process
        """
        pass


class RandomSelector(DCSelector):
    """Random DC ordering"""
    
    def get_dc_order(self, dcs, active_requests, topology):
        dc_ids = [dc.id for dc in dcs]
        np.random.shuffle(dc_ids)
        return dc_ids


class PrioritySelector(DCSelector):
    """Priority-based DC ordering using estimated delay"""
    
    def get_dc_order(self, dcs, active_requests, topology):
        if not active_requests:
            return [dc.id for dc in dcs]
        
        # Find most urgent request (min slack time)
        urgent_req = min(active_requests, key=lambda r: r.max_delay - r.elapsed_time)
        
        # Score DCs based on proximity to source/destination of urgent request
        scores = {}
        for dc in dcs:
            # Base score
            score = 0.0
            
            # Bonus for being the source
            if dc.id == urgent_req.source:
                score += 1000.0
            
            # Penalty based on delay to source and destination
            # Sử dụng get_estimated_delay từ TopologyManager mới
            delay_to_src = topology.get_estimated_delay(urgent_req.source, dc.id)
            delay_to_dst = topology.get_estimated_delay(dc.id, urgent_req.destination)
            
            if delay_to_src != float('inf') and delay_to_dst != float('inf'):
                score += 500.0 - (delay_to_src + delay_to_dst) * 10
            else:
                score -= 1000.0 # Unreachable
            
            scores[dc.id] = score
        
        # Sort by score descending
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class VAESelector(DCSelector):
    """VAE-based DC ordering using learned value function"""
    
    def __init__(self, vae_agent):
        """
        Args:
            vae_agent: Trained VAEAgent instance
        """
        self.agent = vae_agent
    
    def get_dc_order(self, dcs, active_requests, topology):
        from envs.observer import Observer
        
        if not active_requests:
            return [dc.id for dc in dcs]
        
        # Get DC states (Batch inference)
        dc_states = Observer.get_all_dc_states(dcs, active_requests)
        
        # Predict values using VAEAgent
        # predict_dc_values trả về numpy array các giá trị thực
        values = self.agent.predict_dc_values(dc_states)
        
        # Sort by value (descending)
        # values là mảng 1D, argsort trả về indices tăng dần -> [::-1] để giảm dần
        ranked_indices = np.argsort(values.flatten())[::-1]
        
        # Map indices back to DC IDs
        # Lưu ý: dcs phải được đảm bảo thứ tự tương ứng với dc_states
        return [dcs[i].id for i in ranked_indices]