import numpy as np
import config
from core.request import SFCRequest

class SFC_Manager:
    def __init__(self):
        self.all_requests = []
        self.request_ptr = 0
        
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.current_time = 0.0

    def load_requests(self, request_data_list):
        self.all_requests = []
        sorted_data = sorted(request_data_list, key=lambda x: x['arrival_time'])
        
        for req_data in sorted_data:
            req = SFCRequest(
                req_id=req_data['id'],
                vnf_chain=req_data['vnf_chain'],
                source=req_data['source'],
                destination=req_data['destination'],
                arrival_time=req_data['arrival_time'],
                bandwidth=req_data['bandwidth'],
                max_delay=req_data['max_delay'],
                sfc_type=req_data.get('type', 'Unknown')
            )
            self.all_requests.append(req)
        self.request_ptr = 0

    def reset_history(self):
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.current_time = 0.0
        self.request_ptr = 0

    def activate_requests_at_time(self, time):
        """Activate requests with arrival_time <= time"""
        while (self.request_ptr < len(self.all_requests) and 
               self.all_requests[self.request_ptr].arrival_time <= time):
            self.active_requests.append(self.all_requests[self.request_ptr])
            self.request_ptr += 1

    def step(self, time_step_increment):
        self.current_time += time_step_increment
        self.activate_requests_at_time(self.current_time)
        
        for req in self.active_requests:
            req.update_time(time_step_increment)
        
        self._clean_requests()

    def _clean_requests(self):
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
        total_processed = len(self.completed_history) + len(self.dropped_history)
        total_generated = len(self.all_requests)
        
        if total_processed == 0:
            return {
                'acceptance_ratio': 0, 
                'avg_e2e_delay': 0,
                'total_accepted': 0,
                'total_dropped': 0,
                'total_generated': total_generated,
                'current_active': len(self.active_requests)
            }
        
        accepted = len(self.completed_history)
        acc_ratio = (accepted / total_processed) * 100
        
        avg_e2e = np.mean([r.get_total_e2e_delay() for r in self.completed_history]) if self.completed_history else 0.0
        
        return {
            'acceptance_ratio': acc_ratio,
            'total_accepted': accepted,
            'total_dropped': len(self.dropped_history),
            'total_generated': total_generated,
            'avg_e2e_delay': avg_e2e,
            'current_active': len(self.active_requests)
        }