from core.request import Request

class SFCManager:
    '''
    Quản lý vòng đời tất cả các request
    '''
    def __init__(self):
        self.all_requests = []        
        self.active_requests = []
        self.request_cursor = 0

        # History lists cho thống kê
        self.completed_history = []
        self.dropped_history = []

    def load(self, requests_data):
        '''Load và sắp xếp requests theo arrival_time'''
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
        '''Kích hoạt request có arrival_time <= thời gian hiện tại'''
        while self.request_cursor < len(self.all_requests):
            req = self.all_requests[self.request_cursor]
            if req.arrival_time <= current_sim_time:
                self.active_requests.append(req)
                self.request_cursor += 1
            else:
                break
        
    def step(self, dt):
        '''
        Cập nhật thời gian cho các request active và dọn dẹp request đã xong/hỏng.
        '''
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
        """Trả về thống kê hiện tại cho Environment."""
        total_completed = len(self.completed_history)
        total_dropped = len(self.dropped_history)
        total_processed = total_completed + total_dropped
        
        acc_ratio = (total_completed / total_processed * 100.0) if total_processed > 0 else 0.0
        
        avg_delay = 0.0
        if total_completed > 0:
            avg_delay = sum(r.elapsed_time for r in self.completed_history) / total_completed
            
        return {
            'acceptance_ratio': acc_ratio,
            'avg_e2e_delay': avg_delay,
            'total_generated': self.request_cursor,
            'active_count': len(self.active_requests),
            'completed_count': total_completed,
            'dropped_count': total_dropped
        }