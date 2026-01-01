import numpy as np
import config
from core.request import SFCRequest

class SFC_Manager:
    """Quản lý vòng đời của tất cả SFC requests dựa trên Arrival Time"""
    
    def __init__(self):
        self.all_requests = []       # Danh sách gốc (đã sắp xếp theo thời gian)
        self.request_ptr = 0         # Con trỏ để trỏ tới request tiếp theo sẽ đến
        
        self.active_requests = []    # Request đang trong hệ thống (đang chờ đặt hoặc đang xử lý)
        self.completed_history = []  # Lịch sử hoàn thành
        self.dropped_history = []    # Lịch sử thất bại
        self.current_time = 0.0

    def load_requests(self, request_data_list):
        """Load và sắp xếp request theo thời gian đến"""
        self.all_requests = []
        # Chuyển đổi data và sắp xếp
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
        self.request_ptr = 0

    def reset_history(self):
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.current_time = 0.0
        self.request_ptr = 0

    def step(self, time_step_increment):
        """
        Tiến triển thời gian của toàn bộ hệ thống.
        1. Kích hoạt request mới đến.
        2. Cập nhật thời gian cho các request đang active.
        3. Dọn dẹp request đã xong/bị hủy.
        """
        self.current_time += time_step_increment
        
        # 1. Kích hoạt request mới dựa trên Arrival Time (Sử dụng con trỏ để tối ưu)
        while (self.request_ptr < len(self.all_requests) and 
               self.all_requests[self.request_ptr].arrival_time <= self.current_time):
            
            new_req = self.all_requests[self.request_ptr]
            self.active_requests.append(new_req)
            self.request_ptr += 1

        # 2. Cập nhật thời gian và trạng thái cho từng request đang active
        for req in self.active_requests:
            # Truyền bước thời gian vào để request tự cập nhật elapsed_time và logic completion
            req.update_time(time_step_increment)

        # 3. Dọn dẹp
        self.clean_requests()

    def clean_requests(self):
        """Phân loại request đã kết thúc vào lịch sử"""
        still_active = []
        for req in self.active_requests:
            if req.is_completed:
                self.completed_history.append(req)
            elif req.is_dropped:
                self.dropped_history.append(req)
            else:
                still_active.append(req)
        self.active_requests = still_active

    def get_pending_placement_requests(self):
        """Trả về danh sách các request đang active nhưng chưa đặt xong VNF"""
        return [req for req in self.active_requests if not req.is_placed and not req.is_dropped]

    def get_statistics(self):
        """Tính toán thống kê hiệu năng"""
        total_processed = len(self.completed_history) + len(self.dropped_history)
        if total_processed == 0:
            return {'acceptance_ratio': 0, 'avg_e2e_delay': 0}
            
        accepted = len(self.completed_history)
        acc_ratio = (accepted / total_processed) * 100
        
        avg_e2e = 0.0
        if self.completed_history:
            avg_e2e = np.mean([r.get_current_e2e_delay() for r in self.completed_history])
        
        return {
            'acceptance_ratio': acc_ratio,
            'total_accepted': accepted,
            'total_dropped': len(self.dropped_history),
            'avg_e2e_delay': avg_e2e,
            'current_active': len(self.active_requests)
        }