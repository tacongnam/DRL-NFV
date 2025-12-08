import numpy as np
import config
import spaces.request as request

class SFC_Manager:
    def __init__(self):
        self.active_requests = [] # Các request đang được xử lý hoặc chờ
        self.request_counter = 0

        self.completed_history = []
        self.dropped_history = []

    def generate_requests(self, time_step, num_dcs): # Thêm num_dcs
        new_reqs = []
        for s_type in config.SFC_TYPES:
             if np.random.rand() < 0.1: 
                min_b, max_b = config.SFC_SPECS[s_type]['bundle']
                count = np.random.randint(min_b // 10, max_b // 10 + 1)
                for _ in range(count):
                    src = np.random.randint(0, num_dcs) # Dùng num_dcs thay vì config.NUM_DCS
                    dst = np.random.randint(0, num_dcs)
                    while dst == src: dst = np.random.randint(0, num_dcs)
                    
                    req = request(self.request_counter, s_type, src, dst, time_step)
                    new_reqs.append(req)
                    self.request_counter += 1
        
        self.active_requests.extend(new_reqs)
        return len(new_reqs)

    def clean_requests(self):
        """Xóa các request đã xong hoặc bị drop"""
        completed = [r for r in self.active_requests if r.is_completed]
        dropped = [r for r in self.active_requests if r.is_dropped]
        
        self.completed_history.extend(completed)
        self.dropped_history.extend(dropped)

        self.active_requests = [r for r in self.active_requests 
                                if not r.is_completed and not r.is_dropped]

    def reset_history(self):
        """Reset stats"""
        self.completed_history = []
        self.dropped_history = []
        
    def get_global_state_info(self):
        """Tạo Input 3 cho DRL"""
        # Matrix [|S| x (4 + |V|)]
        state = np.zeros((config.NUM_SFC_TYPES, 4 + config.NUM_VNF_TYPES))
        
        for i, s_type in enumerate(config.SFC_TYPES):
            reqs = [r for r in self.active_requests if r.type == s_type]
            count = len(reqs)
            if count > 0:
                avg_rem_time = np.mean([r.max_delay - r.elapsed_time for r in reqs])
                bw_req = config.SFC_SPECS[s_type]['bw']
                
                # Cột 0: Request Count
                state[i, 0] = count
                # Cột 1: Min/Avg Remaining Time
                state[i, 1] = avg_rem_time
                # Cột 2: BW Requirement
                state[i, 2] = bw_req
                # Cột 3: Pending VNFs count (Total)
                pending_vnfs = 0
                vnf_counts = np.zeros(config.NUM_VNF_TYPES)
                
                for r in reqs:
                    next_v = r.get_next_vnf()
                    if next_v:
                        pending_vnfs += 1
                        v_idx = config.VNF_TYPES.index(next_v)
                        vnf_counts[v_idx] += 1
                
                state[i, 3] = pending_vnfs
                # Cột 4 đến hết: Specific VNF pending counts
                state[i, 4:] = vnf_counts
                
        return state.flatten()