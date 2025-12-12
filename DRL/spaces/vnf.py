# spaces/vnf.py
import config

class VNFInstance:
    """Đại diện cho một VNF instance đã được cài đặt tại DC"""
    
    def __init__(self, vnf_type, dc_id):
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.remaining_proc_time = 0.0  # ms
        self.assigned_sfc_id = None
        self.waiting_time = 0.0  # ms - thời gian chờ trước khi được cấp phát

    def is_idle(self):
        """Kiểm tra VNF có đang rảnh không"""
        return self.remaining_proc_time <= 0 and self.assigned_sfc_id is None

    def assign(self, sfc_id, proc_time_ms, waiting_time_ms=0.0):
        """
        Cấp phát VNF cho một SFC request
        
        Args:
            sfc_id: ID của SFC request
            proc_time_ms: Thời gian xử lý (ms)
            waiting_time_ms: Thời gian chờ trước khi xử lý (ms)
        """
        self.assigned_sfc_id = sfc_id
        # Convert proc_time to ticks (làm tròn lên để đảm bảo có thời gian xử lý)
        self.remaining_proc_time = max(config.TIME_STEP, proc_time_ms)
        self.waiting_time = waiting_time_ms

    def tick(self):
        """Cập nhật trạng thái VNF sau mỗi time step"""
        if self.remaining_proc_time > 0:
            self.remaining_proc_time = max(0, self.remaining_proc_time - config.TIME_STEP)
            
            # Khi hoàn thành xử lý, giải phóng VNF
            if self.remaining_proc_time <= 0:
                self.assigned_sfc_id = None
                self.waiting_time = 0.0
                self.remaining_proc_time = 0.0

    def get_total_delay(self):
        """Trả về tổng delay (waiting + processing)"""
        return self.waiting_time + self.remaining_proc_time