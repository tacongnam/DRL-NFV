# spaces/request.py
import config

class SFCRequest:
    """Đại diện cho một SFC request"""
    
    def __init__(self, req_id, sfc_type, source, destination, arrival_time):
        self.id = req_id
        self.type = sfc_type
        self.specs = config.SFC_SPECS[sfc_type]
        self.chain = self.specs['chain'][:]  # VNF chain
        self.current_vnf_index = 0
        
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        
        self.max_delay = self.specs['delay']  # ms
        self.elapsed_time = 0.0  # ms
        
        self.is_dropped = False
        self.is_completed = False
        self.all_vnfs_processed = False  # Flag để đánh dấu đã xử lý xong
        
        # Lưu thông tin VNF đã được đặt: [(vnf_name, dc_id, propagation_delay, processing_delay)]
        self.placed_vnfs = []
        
        # Tổng delay tích lũy
        self.total_propagation_delay = 0.0  # ms
        self.total_processing_delay = 0.0   # ms
        
        # Track VNF instances đang xử lý request này
        self.processing_vnf_instances = []  # List of VNFInstance objects

    def get_next_vnf(self):
        """Lấy VNF tiếp theo trong chain cần được đặt"""
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None

    def advance_chain(self, dc_id, prop_delay=0.0, proc_delay=0.0, vnf_instance=None):
        """
        Tiến bộ trong chain sau khi đặt thành công VNF
        
        Args:
            dc_id: ID của DC nơi VNF được đặt
            prop_delay: Propagation delay từ DC trước đến DC này (ms)
            proc_delay: Processing delay của VNF này (ms)
            vnf_instance: VNFInstance object đang xử lý
        """
        if self.is_completed:
            return
        
        vnf_name = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf_name, dc_id, prop_delay, proc_delay))
        
        # Cộng dồn delay
        self.total_propagation_delay += prop_delay
        self.total_processing_delay += proc_delay
        
        # Track VNF instance
        if vnf_instance:
            self.processing_vnf_instances.append(vnf_instance)
        
        self.current_vnf_index += 1
        
        # Kiểm tra đã đặt hết VNF chưa (nhưng chưa xử lý xong)
        if self.current_vnf_index >= len(self.chain):
            # Đã đặt hết VNF, chờ xử lý
            pass

    def check_completion(self):
        """
        Kiểm tra xem tất cả VNF đã xử lý xong chưa
        Chỉ hoàn thành khi:
        1. Đã đặt hết VNF trong chain
        2. Tất cả VNF đã xử lý xong (idle)
        """
        if self.is_completed or self.is_dropped:
            return
        
        # Đã đặt hết VNF chưa?
        if self.current_vnf_index >= len(self.chain):
            # Kiểm tra tất cả VNF đã xử lý xong chưa
            all_idle = all(vnf.is_idle() for vnf in self.processing_vnf_instances)
            
            if all_idle:
                self.is_completed = True
                self.all_vnfs_processed = True

    def get_total_e2e_delay(self):
        """Tính tổng E2E delay (propagation + processing)"""
        return self.total_propagation_delay + self.total_processing_delay

    def get_remaining_time(self):
        """Thời gian còn lại trước khi bị drop"""
        return max(0, self.max_delay - self.elapsed_time)

    def update_time(self):
        """Cập nhật thời gian đã trôi qua"""
        if self.is_completed or self.is_dropped:
            return
        
        self.elapsed_time += config.TIME_STEP
        
        # Kiểm tra completion trước khi check drop
        self.check_completion()
        
        # Kiểm tra điều kiện drop (chỉ drop nếu chưa hoàn thành)
        if not self.is_completed and self.elapsed_time > self.max_delay:
            self.is_dropped = True

    def get_last_placed_dc(self):
        """Lấy DC ID của VNF cuối cùng được đặt"""
        if self.placed_vnfs:
            return self.placed_vnfs[-1][1]
        return None