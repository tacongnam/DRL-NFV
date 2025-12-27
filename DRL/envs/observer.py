import numpy as np
from DRL import config

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
        # Sử dụng phép chia numpy vector sẽ nhanh hơn chia từng biến
        res_state = np.array([
            dc.cpu / config.DC_CPU_CYCLES,
            dc.ram / config.DC_RAM,
            dc.storage / config.DC_STORAGE
        ], dtype=np.float32)
        
        # 2. VNF Counts
        # Map đếm số lượng VNF (đoạn này vẫn phải duyệt list VNF của DC, 
        # nhưng thường số VNF trên 1 DC ít hơn nhiều so với số Request toàn mạng)
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        for vnf in dc.installed_vnfs:
            idx = vnf_map.get(vnf.vnf_type, 0)
            installed_counts[idx] += 1
            if vnf.is_idle():
                idle_counts[idx] += 1
        
        # Normalize counts (chia cho 10.0 hoặc hằng số phù hợp)
        installed_counts /= 10.0
        idle_counts /= 10.0
        
        # 3. SFC Info (Tra cứu O(1) từ global_stats)
        sfc_source_count = 0.0
        min_remaining_time = 1.0 # Default normalized value (~100ms)
        total_bw_need = 0.0
        
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            # Normalize
            sfc_source_count = min(st['source_count'] / 10.0, 1.0)
            
            # Min time: 9999 -> 1.0, 0 -> 0.0
            raw_min_time = st['min_time']
            if raw_min_time > 100.0: raw_min_time = 100.0
            min_remaining_time = raw_min_time / 100.0
            
            total_bw_need = st['bw_sum'] / 1000.0 # Normalize 1Gbps
            
        # Combine
        state = np.concatenate([
            res_state,
            installed_counts,
            idle_counts,
            [sfc_source_count, min_remaining_time, total_bw_need]
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
        """
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)

        # --- S1: DC State (Shape: 2*V + 2) ---
        # Chỉ lấy CPU, RAM (bỏ Storage) để khớp shape model DRL
        res_state = np.array([
            dc.cpu / config.DC_CPU_CYCLES,
            dc.ram / config.DC_RAM
        ], dtype=np.float32)
        
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        for vnf in dc.installed_vnfs:
            idx = vnf_map.get(vnf.vnf_type, 0)
            installed_counts[idx] += 1
            if vnf.is_idle(): idle_counts[idx] += 1
        
        installed_counts /= 10.0
        idle_counts /= 10.0
        
        s1 = np.concatenate([res_state, installed_counts, idle_counts], dtype=np.float32)

        # --- S2: DC-Request State (Simplified without SFC types) ---
        # For each VNF type, count how many requests need it at this DC
        s2_list = []
        for vnf_type in config.VNF_TYPES:
            # Count requests that need this VNF type and originate from this DC
            type_count = sum(1 for r in active_reqs 
                           if r.source == dc.id and vnf_type in r.chain)
            s2_list.append(min(type_count / 10.0, 1.0))
        
        # Pad to match expected size (originally NUM_SFC_TYPES * (1 + 2*NUM_VNF_TYPES))
        # Simplify to just NUM_VNF_TYPES features
        while len(s2_list) < config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES):
            s2_list.append(0.0)
        
        s2 = np.array(s2_list[:config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES)], dtype=np.float32)

        # --- S3: Global Request State (Simplified) ---
        s3_list = []
        
        # Overall statistics
        total_count = len(active_reqs) / 50.0
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if active_reqs else 0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if active_reqs else 0
        
        # Calculate drop rate (overall, not per type)
        total_finished = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        dropped_rate = len(sfc_manager.dropped_history) / total_finished if total_finished > 0 else 0.0
        
        # Per VNF type demand
        vnf_demand = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        for req in active_reqs:
            for vnf_type in req.chain:
                vnf_demand[vnf_type] += 1
        vnf_demand /= max(1, len(active_reqs))
        
        # Repeat pattern to match expected size
        for _ in range(config.NUM_SFC_TYPES):
            s3_list.extend([total_count, avg_rem, avg_bw, dropped_rate])
            s3_list.extend(vnf_demand)
            
        s3 = np.array(s3_list[:config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES)], dtype=np.float32)

        return (s1, s2, s3)