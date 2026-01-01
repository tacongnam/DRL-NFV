import numpy as np
import config

class Observer:
    """
    Observer được tối ưu hóa cho GenAI:
    - Sử dụng pre-computation để tránh lặp lại vòng lặp request (O(N) -> O(1)).
    - Cung cấp đầy đủ các method cần thiết cho Environment và Model.
    """

    @staticmethod
    def get_state_dim():
        """
        Trả về kích thước vector trạng thái (Input dimension cho VAE).
        Cấu trúc:
        - Resources (3): CPU, RAM, Storage
        - Installed VNFs (NUM_VNF_TYPES)
        - Idle VNFs (NUM_VNF_TYPES)
        - SFC Info (3): Source Count, Min Remaining Time, BW Need
        """
        return 3 + (2 * config.NUM_VNF_TYPES) + 3

    @staticmethod
    def get_active_requests(sfc_manager):
        """Helper để lấy list active request (đã lọc)"""
        return [r for r in sfc_manager.active_requests 
                if not r.is_completed and not r.is_dropped]

    @staticmethod
    def precompute_global_stats(sfc_manager, active_reqs=None):
        """
        [OPTIMIZATION]
        Tính toán trước các chỉ số thống kê từ danh sách request.
        Giúp giảm độ phức tạp từ O(Num_DC * Num_Req) xuống O(Num_Req).
        
        Args:
            sfc_manager: Quản lý SFC
            active_reqs: List request đã lọc (optional)
            
        Returns:
            dict: Map {dc_id: {stats...}}
        """
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)
        
        stats_map = {}
        
        for req in active_reqs:
            sid = req.source
            
            # Init entry nếu chưa có
            if sid not in stats_map:
                stats_map[sid] = {
                    'source_count': 0,
                    'urgency_sum': 0.0,
                    'bw_sum': 0.0,
                    'min_time': 9999.0  # Giá trị khởi tạo lớn
                }
            
            # Cộng dồn
            entry = stats_map[sid]
            entry['source_count'] += 1
            
            # Urgency (dùng cho calculate_value)
            remaining = req.get_remaining_time()
            if remaining < entry['min_time']:
                entry['min_time'] = remaining
            
            entry['urgency_sum'] += 1.0 / (remaining + 1.0)
            
            # BW Need (dùng cho state observation)
            # Lưu ý: Đây là tổng BW request, dùng làm proxy cho "nhu cầu mạng"
            entry['bw_sum'] += req.specs['bw']
            
        return stats_map

    @staticmethod
    def get_dc_state(dc, sfc_manager, global_stats=None):
        """
        Trích xuất trạng thái của 1 DC (Optimized).
        
        Args:
            dc: DataCenter object
            sfc_manager: SFC_Manager object
            global_stats: Dict trả về từ precompute_global_stats (nếu None sẽ tự tính, nhưng chậm)
        """

        # 1. Resources (Normalized)
        # Kiểm tra nếu là server thì tính toán, nếu là switch (is_server=False) thì trả về mảng 0
        if dc.is_server == True:
            # Dùng getattr để an toàn nếu thuộc tính is_server chưa được định nghĩa
            res_state = np.array([
                (dc.cpu / config.DC_CPU_CYCLES) if dc.cpu is not None else 0.0,
                (dc.ram / config.DC_RAM) if dc.ram is not None else 0.0,
                (dc.storage / config.DC_STORAGE) if dc.storage is not None else 0.0
            ], dtype=np.float32)
            
            # 2. VNF Counts (Chỉ chạy cho Server)
            vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
            installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            
            for vnf in dc.installed_vnfs:
                idx = vnf_map.get(vnf.vnf_type, 0)
                installed_counts[idx] += 1
                if vnf.is_idle():
                    idle_counts[idx] += 1
            
            # Normalize counts
            installed_counts /= 10.0
            idle_counts /= 10.0
        else:
            # Đối với Node Switch: Tài nguyên và VNF đều bằng 0
            res_state = np.zeros(3, dtype=np.float32)
            installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)

        # 3. SFC Info (Vẫn lấy cho cả Switch vì cần thông tin Băng thông/SFC đi qua)
        sfc_source_count = 0.0
        min_remaining_time = 1.0 
        total_bw_need = 0.0
        
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            sfc_source_count = min(st.get('source_count', 0) / 10.0, 1.0)
            
            raw_min_time = st.get('min_time', 100.0)
            if raw_min_time > 100.0: raw_min_time = 100.0
            min_remaining_time = raw_min_time / 100.0
            
            total_bw_need = st.get('bw_sum', 0) / 1000.0 
            
        # Combine
        state = np.concatenate([
            res_state,
            installed_counts,
            idle_counts,
            np.array([sfc_source_count, min_remaining_time, total_bw_need], dtype=np.float32)
        ], dtype=np.float32)

        
        return state

    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        """
        Tính Value heuristic (Optimized).
        Target cho Value Network học.
        """
        value = 0.0
        
        # Factor 1 & 3: Urgency & Source Count (Tra cứu O(1))
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            
            # Càng nhiều request gấp (urgency cao) -> Value càng cao
            value += st['urgency_sum'] * 10.0
            
            # Ưu tiên DC là source của nhiều request
            value += st['source_count'] * 15.0
        
        # Factor 2: Resource Availability (Khuyến khích chọn DC còn trống)
        cpu_avail = dc.cpu / config.DC_CPU_CYCLES
        value += cpu_avail * 5.0
        
        return value

    @staticmethod
    def get_all_dc_states(dcs, sfc_manager):
        """
        Lấy state của tất cả DC (Dùng cho Inference/Selector).
        Tự động precompute để tối ưu.
        """
        global_stats = Observer.precompute_global_stats(sfc_manager)
        states = []
        for dc in dcs:
            s = Observer.get_dc_state(dc, sfc_manager, global_stats)
            states.append(s)
        return np.array(states, dtype=np.float32)
    
    @staticmethod
    def get_drl_observation(dc, sfc_manager, active_reqs=None):
        """
        [DRL MODE] Trả về Tuple (s1, s2, s3) cho Multi-Input DQN.
        Hỗ trợ cả Server Node và Switch Node.
        """
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)

        # Xác định loại node để tránh truy cập thuộc tính lỗi
        is_server = getattr(dc, 'is_server', True)

        # --- S1: DC State (Shape: 2 + 2*V) ---
        # Sử dụng giá trị mặc định 0.0 nếu dc.cpu/dc.ram là None hoặc không tồn tại (đối với Switch)
        cpu_val = getattr(dc, 'cpu', 0.0) if getattr(dc, 'cpu', 0.0) is not None else 0.0
        ram_val = getattr(dc, 'ram', 0.0) if getattr(dc, 'ram', 0.0) is not None else 0.0

        res_state = np.array([
            cpu_val / config.DC_CPU_CYCLES,
            ram_val / config.DC_RAM
        ], dtype=np.float32)
        
        # Khởi tạo counts bằng 0
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        # Chỉ duyệt VNF nếu là Server và có danh sách installed_vnfs
        if is_server and hasattr(dc, 'installed_vnfs'):
            for vnf in dc.installed_vnfs:
                idx = vnf_map.get(vnf.vnf_type, 0)
                installed_counts[idx] += 1
                if vnf.is_idle(): 
                    idle_counts[idx] += 1
            
            # Normalize
            installed_counts /= 10.0
            idle_counts /= 10.0
        
        # Kết hợp lại, Switch sẽ có res_state và counts đều bằng 0 nhưng đúng shape
        s1 = np.concatenate([res_state, installed_counts, idle_counts], dtype=np.float32)

        # --- S2: DC-Request State ---
        s2_list = []
        # Node Switch vẫn có thể là 'source' (điểm bắt đầu của luồng traffic)
        for vnf_type in config.VNF_TYPES:
            type_count = sum(1 for r in active_reqs 
                        if r.source == dc.id and vnf_type in r.chain)
            s2_list.append(min(type_count / 10.0, 1.0))
        
        # Padding để giữ nguyên shape cho model DQN
        target_s2_size = config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES)
        if len(s2_list) < target_s2_size:
            s2_list.extend([0.0] * (target_s2_size - len(s2_list)))
        
        s2 = np.array(s2_list[:target_s2_size], dtype=np.float32)

        # --- S3: Global Request State ---
        s3_list = []
        
        total_count = len(active_reqs) / 50.0
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if active_reqs else 1.0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if active_reqs else 0.0
        
        total_finished = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        dropped_rate = len(sfc_manager.dropped_history) / total_finished if total_finished > 0 else 0.0
        
        vnf_demand = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        if active_reqs:
            for req in active_reqs:
                for vnf_type in req.chain:
                    # Giả định req.chain chứa index hoặc tên loại VNF
                    idx = vnf_map.get(vnf_type, 0) if isinstance(vnf_type, str) else vnf_type
                    if idx < config.NUM_VNF_TYPES:
                        vnf_demand[idx] += 1
            vnf_demand /= max(1, len(active_reqs))
        
        target_s3_size = config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES)
        for _ in range(config.NUM_SFC_TYPES):
            s3_list.extend([total_count, avg_rem, avg_bw, dropped_rate])
            s3_list.extend(vnf_demand)
                
        s3 = np.array(s3_list[:target_s3_size], dtype=np.float32)

        return (s1, s2, s3)