from core import Request
from core.statistics import Statistics

class SFCManager:
    def __init__(self):
        self.all_requests = []        
        self.active_requests = []
        self.request_cursor = 0
        self.completed_history = []
        self.dropped_history = []

    def load(self, requests_data):
        self.all_requests = []
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.request_cursor = 0

        for r in requests_data:
            req = Request(
                id=r['id'],
                arrival_time=r['arrival_time'],
                source=r['source'],
                destination=r['destination'],
                bandwidth=r['bandwidth'],
                vnf_chain=r['vnf_chain'],
                max_delay=r['max_delay']
            )
            self.all_requests.append(req)

        self.all_requests.sort(key=lambda x: x.arrival_time)
    
    def activate_new_requests(self, current_sim_time):
        while self.request_cursor < len(self.all_requests):
            req = self.all_requests[self.request_cursor]
            if req.arrival_time <= current_sim_time:
                self.active_requests.append(req)
                self.request_cursor += 1
            else:
                break
        
    def step(self, dt):
        still_active = []
        
        for req in self.active_requests:
            req.update_time(dt)

            if req.is_completed:
                self.completed_history.append(req)
            elif req.is_dropped:
                self.dropped_history.append(req)
            else:
                still_active.append(req)
        
        self.active_requests = still_active

    def get_statistics(self):
        stats = Statistics.calculate(
            self.completed_history,
            self.dropped_history,
            self.request_cursor
        )
        stats['active_count'] = len(self.active_requests)
        return stats