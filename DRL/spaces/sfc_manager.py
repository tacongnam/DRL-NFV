# spaces/sfc_manager.py
import numpy as np
import config
from spaces.request import SFCRequest

class SFC_Manager:
    def __init__(self):
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0

    def reset_history(self):
        """Reset trạng thái cho episode mới"""
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0

    def generate_requests(self, time_step, num_dcs):
        """Sinh request mới"""
        generated_count = 0
        for s_type in config.SFC_TYPES:
            # Giảm xác suất sinh xuống để tránh quá tải bộ nhớ khi demo
            if np.random.rand() < 0.2: 
                min_b, max_b = config.SFC_SPECS[s_type]['bundle']
                count = np.random.randint(min_b, max_b + 1)
                
                for _ in range(count):
                    # Đảm bảo src != dst
                    src = np.random.randint(0, num_dcs)
                    dst = np.random.randint(0, num_dcs)
                    while dst == src:
                        dst = np.random.randint(0, num_dcs)
                        
                    req = SFCRequest(self.req_counter, s_type, src, dst, time_step)
                    self.active_requests.append(req)
                    self.req_counter += 1
                    generated_count += 1
        return generated_count

    def clean_requests(self):
        """Dọn dẹp request đã hoàn thành hoặc bị drop"""
        active = []
        for r in self.active_requests:
            if r.is_completed:
                self.completed_history.append(r)
            elif r.is_dropped:
                self.dropped_history.append(r)
            else:
                active.append(r)
        self.active_requests = active

    def get_global_state_info(self):
        """Tạo Input 3 cho DRL (Global State)"""
        # Shape: [NUM_SFC_TYPES, 4 + NUM_VNF_TYPES]
        state_matrix = np.zeros((config.NUM_SFC_TYPES, 4 + config.NUM_VNF_TYPES), dtype=np.float32)
        vnf_to_idx = {v: i for i, v in enumerate(config.VNF_TYPES)}
        
        # Gom nhóm request để xử lý nhanh
        reqs_by_type = {t: [] for t in config.SFC_TYPES}
        for r in self.active_requests:
            if not r.is_completed and not r.is_dropped:
                reqs_by_type[r.type].append(r)

        for i, s_type in enumerate(config.SFC_TYPES):
            reqs = reqs_by_type[s_type]
            count = len(reqs)
            
            state_matrix[i, 0] = count
            state_matrix[i, 2] = config.SFC_SPECS[s_type]['bw']
            
            if count > 0:
                # Avg remaining time
                rem_times = [r.max_delay - r.elapsed_time for r in reqs]
                state_matrix[i, 1] = np.mean(rem_times)
                
                # Pending VNFs count
                total_pending = 0
                vnf_counts = np.zeros(config.NUM_VNF_TYPES)
                for r in reqs:
                    next_v = r.get_next_vnf()
                    if next_v:
                        total_pending += 1
                        vnf_counts[vnf_to_idx[next_v]] += 1
                
                state_matrix[i, 3] = total_pending
                state_matrix[i, 4:] = vnf_counts

        return state_matrix.flatten()

    def get_statistics(self):
        """Trả về thống kê hiệu năng (FIXED)"""
        total = self.req_counter
        acc = len(self.completed_history)
        drop = len(self.dropped_history)
        
        acc_ratio = (acc / total * 100) if total > 0 else 0.0
        drop_ratio = (drop / total * 100) if total > 0 else 0.0
        
        return {
            'acceptance_ratio': acc_ratio,
            'drop_ratio': drop_ratio,
            'total_generated': total,
            'total_accepted': acc,
            'total_dropped': drop
        }