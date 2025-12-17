# core/sfc_manager.py
import numpy as np
import config
from core.request import SFCRequest

class SFC_Manager:
    """Quản lý tất cả SFC requests"""
    
    def __init__(self):
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0

    def reset_history(self):
        """Reset về trạng thái ban đầu cho episode mới"""
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0

    def generate_requests(self, time_step, num_dcs):
        """
        Sinh request bundles theo specification
        
        Args:
            time_step: Thời điểm hiện tại (ms)
            num_dcs: Số lượng DC trong mạng
            
        Returns:
            Số lượng request được sinh ra
        """
        generated_count = 0
        
        for sfc_type in config.SFC_TYPES:
            # Random để quyết định có sinh request bundle này không
            if np.random.rand() < 0.3:  # 30% probability
                bundle_min, bundle_max = config.SFC_SPECS[sfc_type]['bundle']
                count = np.random.randint(bundle_min, bundle_max + 1)
                
                for _ in range(count):
                    # Chọn source và destination ngẫu nhiên (khác nhau)
                    src = np.random.randint(0, num_dcs)
                    dst = np.random.randint(0, num_dcs)
                    
                    # Đảm bảo src != dst
                    while dst == src:
                        dst = np.random.randint(0, num_dcs)
                    
                    req = SFCRequest(self.req_counter, sfc_type, src, dst, time_step)
                    self.active_requests.append(req)
                    self.req_counter += 1
                    generated_count += 1
        
        return generated_count

    def clean_requests(self):
        """Di chuyển các request completed/dropped vào history"""
        still_active = []
        
        for req in self.active_requests:
            if req.is_completed:
                self.completed_history.append(req)
            elif req.is_dropped:
                self.dropped_history.append(req)
            else:
                still_active.append(req)
        
        self.active_requests = still_active

    def get_drop_rate(self, sfc_type):
        """
        [NEW] Tính tỷ lệ rớt gói cho một loại SFC cụ thể.
        Được gọi bởi DRL Observer để làm input state.
        
        Returns:
            float: Tỷ lệ rớt (0.0 -> 1.0)
        """
        # Đếm số lượng request thuộc loại sfc_type trong lịch sử
        dropped_count = sum(1 for r in self.dropped_history if r.type == sfc_type)
        completed_count = sum(1 for r in self.completed_history if r.type == sfc_type)
        
        total_finished = dropped_count + completed_count
        
        if total_finished > 0:
            return dropped_count / total_finished
        return 0.0

    def get_statistics(self):
        """
        Tính toán statistics tổng quan
        """
        total = self.req_counter
        accepted = len(self.completed_history)
        dropped = len(self.dropped_history)
        
        # Acc Ratio tính trên tổng request đã sinh ra
        acc_ratio = (accepted / total * 100) if total > 0 else 0.0
        drop_ratio = (dropped / total * 100) if total > 0 else 0.0
        
        # Tính avg E2E delay cho các request đã hoàn thành
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