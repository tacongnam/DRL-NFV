class Request:
    def __init__(self, id, arrival_time, source, destination, bandwidth, vnf_chain, max_delay):
        self.id = id
        self.arrival_time = arrival_time
        self.source = source
        self.destination = destination
        self.bandwidth = bandwidth
        self.chain = vnf_chain  # List[int]: Danh sách các loại VNF yêu cầu
        self.max_delay = max_delay

        # Run time state
        self.current_vnf_idx = 0
        self.elapsed_time = 0.0
        self.placed_vnfs = []   # List[Tuple]: Lưu vết (dc_id, vnf_type)
        self.allocated_paths = [] # List[Tuple]: (u, v, bw, path_nodes) để hoàn trả BW
        
        self.is_completed = False
        self.is_dropped = False

    def get_next_vnf(self):
        """Lấy loại VNF tiếp theo cần đặt. None nếu đã hoàn thành."""
        if not self.is_completed and self.current_vnf_idx < len(self.chain):
            return self.chain[self.current_vnf_idx]
        return None
    
    def get_last_placed_dc(self):
        """Lấy vị trí hiện tại của gói tin (Source hoặc DC vừa đặt VNF)."""
        if not self.placed_vnfs:
            return self.source
        return self.placed_vnfs[-1][0] # (dc_id, vnf_type) -> dc_id
    
    def get_remaining_time(self):
        """Thời gian còn lại trước khi hết hạn."""
        return max(0.0, self.max_delay - self.elapsed_time)

    def advance_chain(self, dc_id, prop_delay, proc_delay, vnf_instance):
        """
        Cập nhật trạng thái khi một VNF được đặt thành công.
        Được gọi từ env._handle_allocation
        """
        if self.is_completed:
            return

        vnf_type = self.chain[self.current_vnf_idx]
        self.placed_vnfs.append((dc_id, vnf_type))
        
        # Cộng dồn thời gian: Truyền dẫn + Xử lý + Chờ (đã tính trong proc_delay nếu có)
        total_delay = prop_delay + proc_delay
        self.elapsed_time += total_delay
        
        self.current_vnf_idx += 1
        
    def check_completion(self):
        """Kiểm tra xem request đã hoàn thành chuỗi chưa."""
        if self.current_vnf_idx >= len(self.chain):
            self.is_completed = True
            return True
        return False

    def update_time(self, dt):
        """Tăng thời gian trôi qua và kiểm tra deadline."""
        if not self.is_completed and not self.is_dropped:
            self.elapsed_time += dt
            if self.elapsed_time > self.max_delay:
                self.is_dropped = True