import numpy as np
import config

class Observer:
    @staticmethod
    def get_state_dim():
        return 3 + (2 * config.MAX_VNF_TYPES) + 3

    @staticmethod
    def get_active_requests(sfc_manager):
        return [r for r in sfc_manager.active_requests 
                if not r.is_completed and not r.is_dropped]

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
    def get_dc_state(dc, sfc_manager, global_stats=None):
        if not dc.is_server:
            res_state = np.zeros(3, dtype=np.float32)
            installed_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        else:
            res_state = np.array([
                (dc.cpu / config.MAX_CPU) if dc.cpu else 0.0,
                (dc.ram / config.MAX_RAM) if dc.ram else 0.0,
                (dc.storage / config.MAX_STORAGE) if dc.storage else 0.0
            ], dtype=np.float32)
            
            res_state = np.clip(res_state, 0.0, 1.0)

            installed_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
            
            for vnf in dc.installed_vnfs:
                vnf_idx = vnf.vnf_type
                if vnf_idx < config.MAX_VNF_TYPES:
                    installed_counts[vnf_idx] += 1
                    if vnf.is_idle():
                        idle_counts[vnf_idx] += 1
            
            installed_counts /= 10.0
            idle_counts /= 10.0

        sfc_source_count = 0.0
        min_remaining_time = 1.0
        total_bw_need = 0.0
        
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            sfc_source_count = min(st.get('source_count', 0) / 10.0, 1.0)
            
            raw_min_time = st.get('min_time', 100.0)
            min_remaining_time = min(raw_min_time / 100.0, 1.0)
            
            total_bw_need = min(st.get('bw_sum', 0) / (config.MAX_BW * 2), 1.0)

        state = np.concatenate([
            res_state,
            installed_counts,
            idle_counts,
            np.array([sfc_source_count, min_remaining_time, total_bw_need], dtype=np.float32)
        ], dtype=np.float32)
        
        return state

    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        """Paper-inspired value: prioritize urgent requests + available resources"""
        value = 0.0
        
        # Factor 1: Request urgency (paper emphasizes E2E delay)
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            # Urgency sum is key metric in paper
            value += st['urgency_sum'] * 20.0
            # Request count at this source
            value += st['source_count'] * 15.0
        
        # Factor 2: Resource availability (paper considers this)
        if dc.is_server:
            cpu_avail = dc.cpu / config.MAX_CPU
            ram_avail = dc.ram / config.MAX_RAM
            storage_avail = dc.storage / config.MAX_STORAGE
            
            # Paper prioritizes DCs with more available resources
            avg_resource = (cpu_avail + ram_avail + storage_avail) / 3.0
            value += avg_resource * 30.0
            
            # Bonus for having idle VNFs (reduces installation overhead)
            idle_count = sum(1 for v in dc.installed_vnfs if v.is_idle())
            value += idle_count * 10.0
        
        return value

    @staticmethod
    def get_all_dc_states(dcs, active_reqs):
        global_stats = Observer.precompute_global_stats(None, active_reqs)
        states = []
        for dc in dcs:
            if dc.is_server:
                s = Observer.get_dc_state(dc, None, global_stats)
                states.append(s)
        return np.array(states, dtype=np.float32)
    
    @staticmethod
    def _encode_chain_pattern(chain, max_length=4):
        chain_seq = np.full(max_length, -1, dtype=np.float32)
        for i, vnf in enumerate(chain[:max_length]):
            if vnf < config.MAX_VNF_TYPES:
                chain_seq[i] = vnf / config.MAX_VNF_TYPES
        
        vnf_presence = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        for vnf in chain:
            if vnf < config.MAX_VNF_TYPES:
                vnf_presence[vnf] = 1.0
        
        return np.concatenate([chain_seq, vnf_presence])
    
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
    def get_drl_observation(dc, sfc_manager, active_reqs=None):
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)

        res_state = np.array([
            (dc.cpu / config.MAX_CPU) if dc.cpu else 0.0,
            (dc.ram / config.MAX_RAM) if dc.ram else 0.0
        ], dtype=np.float32)
        
        installed_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        
        if dc.is_server:
            for vnf in dc.installed_vnfs:
                idx = vnf.vnf_type
                if idx < config.MAX_VNF_TYPES:
                    installed_counts[idx] += 1
                    if vnf.is_idle(): 
                        idle_counts[idx] += 1
            installed_counts /= 10.0
            idle_counts /= 10.0
        
        s1 = np.concatenate([res_state, installed_counts, idle_counts], dtype=np.float32)

        vnf_demand_at_dc = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        dc_chains = []
        for req in active_reqs:
            if req.source == dc.id:
                for vnf_type in req.chain:
                    if isinstance(vnf_type, int) and vnf_type < config.MAX_VNF_TYPES:
                        vnf_demand_at_dc[vnf_type] += 1
                dc_chains.append(req)
        
        vnf_demand_at_dc /= max(1, len(active_reqs))
        chain_patterns_dc = Observer._aggregate_chain_stats(dc_chains, max_top_chains=3)
        s2 = np.concatenate([vnf_demand_at_dc, chain_patterns_dc], dtype=np.float32)

        total_count = len(active_reqs) / 50.0
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if active_reqs else 1.0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if active_reqs else 0.0
        
        total_finished = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        dropped_rate = len(sfc_manager.dropped_history) / total_finished if total_finished > 0 else 0.0
        
        global_vnf_demand = np.zeros(config.MAX_VNF_TYPES, dtype=np.float32)
        if active_reqs:
            for req in active_reqs:
                for vnf_type in req.chain:
                    if isinstance(vnf_type, int) and vnf_type < config.MAX_VNF_TYPES:
                        global_vnf_demand[vnf_type] += 1
            global_vnf_demand /= max(1, len(active_reqs))
        
        chain_patterns_global = Observer._aggregate_chain_stats(active_reqs, max_top_chains=5)
        
        s3 = np.concatenate([
            np.array([total_count, avg_rem, avg_bw, dropped_rate], dtype=np.float32),
            global_vnf_demand,
            chain_patterns_global
        ], dtype=np.float32)

        return (s1, s2, s3)