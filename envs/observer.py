import numpy as np
import config

class Observer:
    @staticmethod
    def get_state_dim():
        # 3(Res) + 20(VNF) + 3(Global) + 1(Connect)
        return 27 + (2 * config.MAX_VNF_TYPES - 20) # Dynamic adjustment

    @staticmethod
    def _get_dc_features(dc):
        """Hàm helper: Trích xuất đặc trưng cơ bản của DC."""
        if not dc.is_server:
            return np.zeros(3), np.zeros(config.MAX_VNF_TYPES), np.zeros(config.MAX_VNF_TYPES)
        
        # Resource
        res = np.array([dc.cpu/config.MAX_CPU, dc.ram/config.MAX_RAM, dc.storage/config.MAX_STORAGE], dtype=np.float32)
        res = np.clip(res, 0.0, 1.0)
        
        # VNFs
        installed, idle = np.zeros(config.MAX_VNF_TYPES), np.zeros(config.MAX_VNF_TYPES)
        for v in dc.installed_vnfs:
            if v.vnf_type < config.MAX_VNF_TYPES:
                installed[v.vnf_type] += 1
                if v.is_idle(): idle[v.vnf_type] += 1
        
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
            cap = sum(g[dc.id][n].get('capacity', 1000) for n in g[dc.id])
            avail = sum(g[dc.id][n].get('bw', 0) for n in g[dc.id])
            conn = avail / cap if cap > 0 else 0.0
        elif not topology: 
            conn = 1.0

        return np.concatenate([res_state, install, idle, [src_count, min_time, bw_need, conn]], dtype=np.float32)

    @staticmethod
    def get_drl_observation(dc, sfc_manager, topology=None, active_reqs=None):
        """State cho DQN (3 nhánh)."""
        if active_reqs is None:
            active_reqs = [r for r in sfc_manager.active_requests if not r.is_completed and not r.is_dropped]

        # Branch 1: DC State
        res_state, install, idle = Observer._get_dc_features(dc)
        s1 = np.concatenate([res_state, install, idle], dtype=np.float32)

        # Branch 2: Demand (Local)
        dc_reqs = [r for r in active_reqs if r.source == dc.id]
        vnf_demand = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        for r in dc_reqs:
            for v in r.chain: 
                if v < config.MAX_VNF_TYPES: vnf_demand[v] += 1
        vnf_demand /= max(1, len(active_reqs))
        s2 = np.concatenate([vnf_demand, Observer._aggregate_chain_stats(dc_reqs, 3)], dtype=np.float32)

        # Branch 3: Global
        total = len(active_reqs)
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if total else 1.0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if total else 0.0
        
        fin = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        drop_rate = len(sfc_manager.dropped_history) / fin if fin else 0.0
        
        glob_demand = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        for r in active_reqs:
            for v in r.chain:
                if v < config.MAX_VNF_TYPES: glob_demand[v] += 1
        glob_demand /= max(1, total)

        s3 = np.concatenate([[total/50.0, avg_rem, avg_bw, drop_rate], glob_demand, Observer._aggregate_chain_stats(active_reqs, 5)], dtype=np.float32)

        return (s1, s2, s3)
    
    @staticmethod
    def precompute_global_stats(sfc_manager, active_reqs=None):
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)
        
        stats_map = {}
        
        for req in active_reqs:
            sid = req.source
            if sid not in stats_map:
                stats_map[sid] = {
                    'source_count': 0,
                    'urgency_sum': 0.0,
                    'bw_sum': 0.0,
                    'min_time': 9999.0
                }
            
            entry = stats_map[sid]
            entry['source_count'] += 1
            
            remaining = req.get_remaining_time()
            if remaining < entry['min_time']:
                entry['min_time'] = remaining
            
            entry['urgency_sum'] += 1.0 / (remaining + 1.0)
            entry['bw_sum'] += req.bandwidth
            
        return stats_map
    
    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        if not dc.is_server:
            return -10.0

        # prev_state format: [CPU, RAM, Stor, (Install*10), (Idle*10), Src, Time, BW, Connectivity]
        # Connectivity là phần tử cuối cùng [-1]
        connectivity_score = prev_state[-1] if prev_state is not None and len(prev_state) > 0 else 0.0

        # Normalize resources
        cpu_score = dc.cpu / config.MAX_CPU
        ram_score = dc.ram / config.MAX_RAM
        resource_val = (cpu_score + ram_score) / 2.0
        
        idle_count = sum(1 for v in dc.installed_vnfs if v.is_idle())
        idle_bonus = min(idle_count / 5.0, 1.0)

        # Chuẩn hóa chi phí (Giả sử max cost tổng ~ 500)
        norm_cost = (dc.cost_c + dc.cost_r + dc.cost_h) / 500.0 
        cost_penalty = min(norm_cost, 1.0)
        
        # Công thức Value cải tiến: Resource + Connectivity - Cost
        final_value = (0.4 * resource_val) + (0.3 * connectivity_score) + (0.3 * idle_bonus) - (0.3 * cost_penalty)
        
        return final_value * 100.0

    @staticmethod
    def _aggregate_chain_stats(active_reqs, max_top_chains=5):
        from collections import Counter
        chain_counter = Counter()
        chain_info = {}
        
        for req in active_reqs:
            chain_tuple = tuple(req.chain)
            chain_counter[chain_tuple] += 1
            if chain_tuple not in chain_info:
                chain_info[chain_tuple] = {'bw_sum': 0.0, 'rem_time_sum': 0.0, 'count': 0}
            chain_info[chain_tuple]['bw_sum'] += req.bandwidth
            chain_info[chain_tuple]['rem_time_sum'] += req.get_remaining_time()
            chain_info[chain_tuple]['count'] += 1
        
        top_chains = chain_counter.most_common(max_top_chains)
        result = []
        
        for chain_tuple, count in top_chains:
            pattern = Observer._encode_chain_pattern(list(chain_tuple))
            info = chain_info[chain_tuple]
            stats = np.array([
                count / max(1, len(active_reqs)),
                info['bw_sum'] / count / 1000.0,
                info['rem_time_sum'] / count / 100.0
            ], dtype=np.float32)
            result.append(np.concatenate([pattern, stats]))
        
        feature_size = 4 + config.MAX_VNF_TYPES + 3
        while len(result) < max_top_chains:
            result.append(np.zeros(feature_size, dtype=np.float32))
        
        return np.concatenate(result)
    
    @staticmethod
    def precompute_global_stats(sfc_manager, active_reqs): # Giữ nguyên logic cũ
        stats = {}
        for req in active_reqs:
            sid = req.source
            if sid not in stats: stats[sid] = {'source_count': 0, 'min_time': 9999, 'bw_sum': 0}
            stats[sid]['source_count'] += 1
            stats[sid]['min_time'] = min(stats[sid]['min_time'], req.get_remaining_time())
            stats[sid]['bw_sum'] += req.bandwidth
        return stats

    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        if not dc.is_server: return -10.0
        # Prev state last idx is connectivity
        conn = prev_state[-1] if len(prev_state) > 0 else 0.0
        
        res = (dc.cpu/config.MAX_CPU + dc.ram/config.MAX_RAM)/2
        idle = min(sum(1 for v in dc.installed_vnfs if v.is_idle())/5.0, 1.0)
        
        norm_cost = min((dc.cost_c + dc.cost_r + dc.cost_h)/500.0, 1.0)
        
        return (0.4 * res + 0.3 * conn + 0.2 * idle - 0.2 * norm_cost) * 100.0

    @staticmethod
    def _aggregate_chain_stats(reqs, top_k): # Giữ nguyên, chỉ thu gọn
        from collections import Counter
        counts = Counter(tuple(r.chain) for r in reqs)
        res = []
        for chain, c in counts.most_common(top_k):
            pat = np.full(4, -1, dtype=np.float32)
            for i, v in enumerate(chain[:4]): pat[i] = v / config.MAX_VNF_TYPES
            pres = np.zeros(config.MAX_VNF_TYPES); pres[list(chain)] = 1.0
            
            # Simplified stats calc
            sub = [r for r in reqs if tuple(r.chain) == chain]
            bw = np.mean([r.bandwidth for r in sub]) / 1000.0
            rem = np.mean([r.get_remaining_time() for r in sub]) / 100.0
            res.append(np.concatenate([pat, pres, [c/len(reqs), bw, rem]]))
            
        feat = 4 + config.MAX_VNF_TYPES + 3
        while len(res) < top_k: res.append(np.zeros(feat, dtype=np.float32))
        return np.concatenate(res)
    
    @staticmethod
    def get_all_dc_states(dcs, active_reqs, topology=None):
        g = Observer.precompute_global_stats(None, active_reqs)
        return np.array([Observer.get_dc_state(d, None, g, topology) for d in dcs if d.is_server], dtype=np.float32)