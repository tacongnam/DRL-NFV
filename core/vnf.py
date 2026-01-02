import config

class VNFInstance:
    """Thể hiện một VNF đang chạy tại một DC."""
    
    def __init__(self, vnf_type, dc_id):
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.remaining_time = 0.0  # Thời gian xử lý còn lại
        self.sfc_id = None         # ID của request đang phục vụ

    def is_idle(self):
        """Kiểm tra VNF có đang rảnh không."""
        return self.remaining_time <= 0 and self.sfc_id is None

    def get_processing_time(self, dc_delay):
        """
        Tính toán thời gian xử lý dự kiến.
        Có thể dựa trên config.VNF_SPECS và dc_delay.
        """
        # Giả sử thời gian xử lý cơ bản + độ trễ nội tại của DC
        base_proc = config.VNF_SPECS[self.vnf_type]['cpu'] * 0.1 # Ví dụ đơn giản
        return base_proc + dc_delay

    def assign(self, sfc_id, dc_delay, waiting_time=0.0):
        """Gán việc cho VNF."""
        self.sfc_id = sfc_id
        # Thời gian bận = Thời gian xử lý + Thời gian chờ (nếu có logic hàng đợi)
        proc_time = self.get_processing_time(dc_delay)
        self.remaining_time = proc_time + waiting_time

    def tick(self):
        """Giảm thời gian xử lý sau mỗi bước mô phỏng."""
        if self.remaining_time > 0:
            self.remaining_time = max(0, self.remaining_time - config.TIME_STEP)
            if self.remaining_time <= 0:
                self.sfc_id = None  # Giải phóng VNF