import numpy as np
import config

class Observer:
    @staticmethod
    def get_state_dim():
        # Dùng cho VAE: Resource(3) + Install(10) + Idle(10) + Global(3) + Connectivity(1)
        # = 27 (với MAX_VNF_TYPES=10)
        return 3 + 2 * config.MAX_VNF_TYPES + 4

    @staticmethod
    def get_active_requests(sfc_manager):
        """Hàm helper bị thiếu gây lỗi AttributeError."""
        return [r for r in sfc_manager.active_requests if not r.is_completed and not r.is_dropped]

    @staticmethod
    def _get_dc_features(dc):
        """Hàm helper: Trích xuất đặc trưng cơ bản của DC."""
        if not dc.is_server:
            return np.zeros(3), np.zeros(config.MAX_VNF_TYPES), np.zeros(config.MAX_VNF_TYPES)
        
        # Resource
        res = np.array([dc.cpu/config.MAX_CPU, dc.ram/config.MAX_RAM, dc.storage/config.MAX_STORAGE], dtype=np.float32)
        res = np.clip(res, 0.0, 1.0)
        
        # VNFs
        installed = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        idle = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        
        for v in dc.installed_vnfs:
            if v.vnf_type < config.MAX_VNF_TYPES:
                installed[v.vnf_type] += 1
                if v.is_idle(): idle[v.vnf_type] += 1
        
        # Normalize (chia cho 10 để scale về khoảng nhỏ)
        return res, installed/10.0, idle/10.0

    @staticmethod
    def get_dc_state(dc, sfc_manager, global_stats=None, topology=None):
        """State cho VAE: Resource + Global + Connectivity"""
        res_state, install, idle = Observer._get_dc_features(dc)

        # Global stats
        src_count, min_time, bw_need = 0.0, 1.0, 0.0
        if global_stats and dc.id in global_stats:
            st = global_stats[dc.id]
            src_count = min(st['source_count'] / 10.0, 1.0)
            min_time = min(st['min_time'] / 100.0, 1.0)
            bw_need = min(st['bw_sum'] / (config.MAX_BW * 2), 1.0)

        # Connectivity
        conn = 0.0
        if topology and dc.id in topology.physical_graph:
            g = topology.physical_graph
            # Tính tổng capacity và bw khả dụng của các cạnh nối với DC này
            cap = 0.0
            avail = 0.0
            for n in g[dc.id]:
                edge = g[dc.id][n]
                cap += edge.get('capacity', 1000)
                avail += edge.get('bw', 0)
            
            conn = avail / cap if cap > 0 else 0.0
        elif not topology: 
            conn = 1.0

        return np.concatenate([res_state, install, idle, [src_count, min_time, bw_need, conn]], dtype=np.float32)

    @staticmethod
    def get_drl_observation(dc, sfc_manager, topology=None, active_reqs=None):
        """State cho DQN (3 nhánh)."""
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)

        # Branch 1: DC State (Shape: 3 + 10 + 10 = 23)
        res_state, install, idle = Observer._get_dc_features(dc)
        s1 = np.concatenate([res_state, install, idle], dtype=np.float32)

        # Branch 2: Demand (Local) (Shape: 10 + 3*17 = 61)
        dc_reqs = [r for r in active_reqs if r.source == dc.id]
        vnf_demand = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        for r in dc_reqs:
            # Chỉ tính next VNF
            nxt = r.get_next_vnf()
            if nxt is not None and nxt < config.MAX_VNF_TYPES:
                vnf_demand[nxt] += 1
        vnf_demand /= max(1, len(active_reqs)) # Normalize
        
        # FIX: _aggregate_chain_stats bây giờ trả về đúng kích thước 17 features mỗi chain
        s2 = np.concatenate([vnf_demand, Observer._aggregate_chain_stats(dc_reqs, 3)], dtype=np.float32)

        # Branch 3: Global (Shape: 4 + 10 + 5*17 = 99)
        total = len(active_reqs)
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if total else 1.0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if total else 0.0
        
        fin = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        drop_rate = len(sfc_manager.dropped_history) / fin if fin else 0.0
        
        glob_demand = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        for r in active_reqs:
            nxt = r.get_next_vnf()
            if nxt is not None and nxt < config.MAX_VNF_TYPES: 
                glob_demand[nxt] += 1
        glob_demand /= max(1, total)

        s3 = np.concatenate([
            [min(total/50.0, 1.0), avg_rem, avg_bw, drop_rate], 
            glob_demand, 
            Observer._aggregate_chain_stats(active_reqs, 5)
        ], dtype=np.float32)

        return (s1, s2, s3)
    
    @staticmethod
    def precompute_global_stats(sfc_manager, active_reqs=None):
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)
        
        stats_map = {}
        for req in active_reqs:
            sid = req.source
            if sid not in stats_map:
                stats_map[sid] = {'source_count': 0, 'bw_sum': 0.0, 'min_time': 9999.0}
            
            entry = stats_map[sid]
            entry['source_count'] += 1
            remaining = req.get_remaining_time()
            if remaining < entry['min_time']: entry['min_time'] = remaining
            entry['bw_sum'] += req.bandwidth
            
        return stats_map
    
    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        if not dc.is_server:
            return -10.0

        # prev_state format: [CPU, RAM, Stor, (Install...), (Idle...), Src, Time, BW, Connectivity]
        # Connectivity là phần tử cuối cùng [-1]
        connectivity_score = prev_state[-1] if prev_state is not None and len(prev_state) > 0 else 0.0

        cpu_score = dc.cpu / config.MAX_CPU
        ram_score = dc.ram / config.MAX_RAM
        resource_val = (cpu_score + ram_score) / 2.0
        
        idle_count = sum(1 for v in dc.installed_vnfs if v.is_idle())
        idle_bonus = min(idle_count / 5.0, 1.0)

        # Chuẩn hóa chi phí (ước lượng max cost ~ 500)
        norm_cost = (dc.cost_c + dc.cost_r + dc.cost_h) / 500.0 
        cost_penalty = min(norm_cost, 1.0)
        
        # Value formula
        final_value = (0.4 * resource_val) + (0.3 * connectivity_score) + (0.3 * idle_bonus) - (0.2 * cost_penalty)
        return final_value * 10.0 # Scale lên 1 chút

    @staticmethod
    def _aggregate_chain_stats(reqs, top_k):
        """
        FIX: Đã thêm lại phần VNF Type One-Hot Encoding để khớp Shape.
        Shape mỗi chain: 4 (Pattern) + 10 (VNF Presence) + 3 (Stats) = 17 features.
        """
        from collections import Counter
        # Chỉ lấy 4 phần tử đầu của chain để làm pattern
        counts = Counter(tuple(r.chain[:4]) for r in reqs)
        res = []
        
        feature_size = 4 + config.MAX_VNF_TYPES + 3 # = 17
        
        for chain_tuple, count in counts.most_common(top_k):
            # 1. Pattern (4)
            pat = np.full(4, -1.0, dtype=np.float32)
            for i, v in enumerate(chain_tuple): 
                if i < 4: pat[i] = v / config.MAX_VNF_TYPES
            
            # 2. Presence (10) - FIX: Đây là phần bị thiếu gây lỗi Shape 31 vs 61
            presence = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
            for v in chain_tuple:
                if v < config.MAX_VNF_TYPES: presence[v] = 1.0
            
            # 3. Stats (3)
            # Lọc các request thuộc chain này để tính avg
            sub_reqs = [r for r in reqs if tuple(r.chain[:4]) == chain_tuple]
            bw = np.mean([r.bandwidth for r in sub_reqs]) / 1000.0
            rem = np.mean([r.get_remaining_time() for r in sub_reqs]) / 100.0
            freq = count / max(1, len(reqs))
            
            stats = np.array([freq, bw, rem], dtype=np.float32)
            
            # Concatenate: 4 + 10 + 3 = 17
            res.append(np.concatenate([pat, presence, stats]))
            
        # Padding nếu không đủ top_k chains
        while len(res) < top_k: 
            res.append(np.zeros(feature_size, dtype=np.float32))
            
        return np.concatenate(res)
    
    @staticmethod
    def get_all_dc_states(dcs, active_reqs, topology=None):
        g = Observer.precompute_global_stats(None, active_reqs)
        return np.array([Observer.get_dc_state(d, None, g, topology) for d in dcs if d.is_server], dtype=np.float32)