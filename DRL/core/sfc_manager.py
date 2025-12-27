# core/sfc_manager.py
import numpy as np
from DRL import config
from DRL.core.request import SFCRequest

class SFC_Manager:
    """Manages all SFC requests loaded from data"""
    
    def __init__(self):
        self.all_requests = []  # All requests loaded from data
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.current_time = 0  # Current simulation time (ms)

    def load_requests(self, request_data_list):
        """
        Load requests from parsed data (from get_R).
        
        Args:
            request_data_list: List of dicts from Read_data.get_R(), each containing:
                - 'id': Request ID
                - 'arrival_time': Arrival time (ms)
                - 'source': Source node ID
                - 'destination': Destination node ID
                - 'vnf_chain': List of VNF type indices
                - 'bandwidth': Bandwidth required
                - 'max_delay': Maximum delay allowed (ms)
        """
        self.all_requests = []
        
        # Sort by arrival time
        sorted_data = sorted(request_data_list, key=lambda x: x['arrival_time'])
        
        for req_data in sorted_data:
            req = SFCRequest(
                req_id=req_data['id'],
                vnf_chain=req_data['vnf_chain'],
                source=req_data['source'],
                destination=req_data['destination'],
                arrival_time=req_data['arrival_time'],
                bandwidth=req_data['bandwidth'],
                max_delay=req_data['max_delay']
            )
            self.all_requests.append(req)

    def reset_history(self):
        """Reset to initial state for new episode"""
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.current_time = 0

    def activate_requests_at_time(self, time_step):
        """
        Activate requests that arrive at current time step.
        
        Args:
            time_step: Current simulation time (ms)
            
        Returns:
            Number of newly activated requests
        """
        self.current_time = time_step
        activated_count = 0
        
        for req in self.all_requests:
            if req.arrival_time == time_step:
                self.active_requests.append(req)
                activated_count += 1
        
        return activated_count

    def clean_requests(self):
        """Move completed/dropped requests to history"""
        still_active = []
        
        for req in self.active_requests:
            if req.is_completed:
                self.completed_history.append(req)
            elif req.is_dropped:
                self.dropped_history.append(req)
            else:
                still_active.append(req)
        
        self.active_requests = still_active

    def get_statistics(self):
        """
        Calculate overall statistics.
        """
        total = len(self.all_requests)
        accepted = len(self.completed_history)
        dropped = len(self.dropped_history)
        
        # Acceptance ratio based on total requests
        acc_ratio = (accepted / total * 100) if total > 0 else 0.0
        drop_ratio = (dropped / total * 100) if total > 0 else 0.0
        
        # Calculate avg E2E delay for completed requests
        avg_e2e = 0.0
        if self.completed_history:
            avg_e2e = np.mean([r.get_total_e2e_delay() for r in self.completed_history])
        
        return {
            'acceptance_ratio': acc_ratio,
            'drop_ratio': drop_ratio,
            'total_generated': total,
            'total_accepted': accepted,
            'total_dropped': dropped,
            'avg_e2e_delay': avg_e2e
        }