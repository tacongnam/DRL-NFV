# spaces/request.py
import config

class SFCRequest:
    """Đại diện cho một yêu cầu SFC (Service Function Chain)"""
    
    def __init__(self, req_id, vnf_chain, source, destination, arrival_time, bandwidth, max_delay):
        self.id = req_id
        self.chain = vnf_chain[:]  
        self.current_vnf_index = 0
        
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        self.bandwidth = bandwidth
        self.max_delay = max_delay  
        
        # Trạng thái
        self.elapsed_time = 0.0     # Thời gian đã trôi qua kể từ khi request đến
        self.is_dropped = False     # Bị hủy (do quá hạn hoặc thiếu tài nguyên)
        self.is_completed = False   # Đã hoàn thành toàn bộ chuỗi và đến đích
        
        # Dữ liệu placement
        self.placed_vnfs = []       # [(vnf_type, dc_id, prop_delay, proc_delay)]
        self.total_propagation_delay = 0.0
        self.total_processing_delay = 0.0
        
        # Quản lý thời gian xử lý riêng cho từng request trên VNF
        # Thay vì check vnf.is_idle(), ta theo dõi thời gian còn lại phải xử lý của request này
        self.remaining_processing_time = 0.0 

    @property
    def is_placed(self):
        """Kiểm tra xem tất cả các VNF đã được đặt vào các node chưa"""
        return self.current_vnf_index >= len(self.chain)

    def get_next_vnf(self):
        if not self.is_placed:
            return self.chain[self.current_vnf_index]
        return None

    def advance_chain(self, dc_id, prop_delay, proc_delay):
        """
        Tiến hành đặt VNF tiếp theo trong chuỗi.
        """
        if self.is_completed or self.is_placed:
            return
        
        vnf_type_idx = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf_type_idx, dc_id, prop_delay, proc_delay))
        
        self.total_propagation_delay += prop_delay
        # Cộng dồn vào tổng thời gian cần xử lý
        self.remaining_processing_time += proc_delay
        self.total_processing_delay += proc_delay
        
        self.current_vnf_index += 1

    def update_time(self, time_step=None):
        """Cập nhật thời gian và trạng thái request"""
        if self.is_completed or self.is_dropped:
            return
        
        step = time_step if time_step is not None else config.TIME_STEP
        self.elapsed_time += step
        
        # 1. Trừ thời gian đang xử lý (nếu đã được đặt vào VNF)
        if self.remaining_processing_time > 0:
            self.remaining_processing_time = max(0, self.remaining_processing_time - step)
        
        # 2. Kiểm tra hoàn thành: 
        # Đã đặt hết VNF VÀ đã xử lý xong thời gian proc_delay
        if self.is_placed and self.remaining_processing_time <= 0:
            # Lưu ý: Cần tính thêm prop_delay chặng cuối về đích ở đây nếu cần
            self.is_completed = True

        # 3. Kiểm tra vi phạm deadline (Latency violation)
        if not self.is_completed and self.elapsed_time > self.max_delay:
            self.is_dropped = True

    def get_current_e2e_delay(self):
        """Tổng delay hiện tại (gồm cả truyền dẫn và xử lý)"""
        return self.total_propagation_delay + self.total_processing_delay

    def get_last_placed_node_id(self):
        """Trả về ID của node cuối cùng trong chuỗi hiện tại (hoặc source nếu chưa đặt)"""
        if not self.placed_vnfs:
            return self.source
        return self.placed_vnfs[-1][1]