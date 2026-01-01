import numpy as np
import config

class Observer:
    @staticmethod
    def get_state_dim():
        """
        VAE State: [cpu, ram, storage, installed_vnfs, idle_vnfs, 
                    source_count, min_rem_time, bw_need]
        """
        return 3 + (2 * config.NUM_VNF_TYPES) + 3

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
        """VAE state - simplified for DC selection"""
        if not dc.is_server:
            res_state = np.zeros(3, dtype=np.float32)
            installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        else:
            res_state = np.array([
                (dc.cpu / config.DC_CPU_CYCLES) if dc.cpu is not None else 0.0,
                (dc.ram / config.DC_RAM) if dc.ram is not None else 0.0,
                (dc.storage / config.DC_STORAGE) if dc.storage is not None else 0.0
            ], dtype=np.float32)
            
            vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
            installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            
            for vnf in dc.installed_vnfs:
                idx = vnf_map.get(vnf.vnf_type, 0)
                installed_counts[idx] += 1
                if vnf.is_idle():
                    idle_counts[idx] += 1
            
            installed_counts /= 10.0
            idle_counts /= 10.0

        sfc_source_count = 0.0
        min_remaining_time = 1.0
        total_bw_need = 0.0
        
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            sfc_source_count = min(st.get('source_count', 0) / 10.0, 1.0)
            
            raw_min_time = st.get('min_time', 100.0)
            if raw_min_time > 100.0:
                raw_min_time = 100.0
            min_remaining_time = raw_min_time / 100.0
            
            total_bw_need = st.get('bw_sum', 0) / 1000.0

        state = np.concatenate([
            res_state,
            installed_counts,
            idle_counts,
            np.array([sfc_source_count, min_remaining_time, total_bw_need], dtype=np.float32)
        ], dtype=np.float32)
        
        return state

    @staticmethod
    def calculate_dc_value(dc, sfc_manager, prev_state, global_stats=None):
        value = 0.0
        
        if global_stats is not None and dc.id in global_stats:
            st = global_stats[dc.id]
            value += st['urgency_sum'] * 10.0
            value += st['source_count'] * 15.0
        
        if dc.is_server:
            cpu_avail = dc.cpu / config.DC_CPU_CYCLES
            value += cpu_avail * 5.0
        
        return value

    @staticmethod
    def get_all_dc_states(dcs, sfc_manager):
        global_stats = Observer.precompute_global_stats(sfc_manager)
        states = []
        for dc in dcs:
            if dc.is_server:
                s = Observer.get_dc_state(dc, sfc_manager, global_stats)
                states.append(s)
        return np.array(states, dtype=np.float32)
    
    @staticmethod
    def _encode_chain_pattern(chain, max_length=4):
        """
        Encode SFC chain pattern into fixed-size representation.
        
        Args:
            chain: List of VNF indices [0, 1, 2, ...]
            max_length: Maximum chain length to support
            
        Returns:
            Array of shape (max_length + NUM_VNF_TYPES,)
            - First max_length elements: chain sequence (padded with -1)
            - Next NUM_VNF_TYPES elements: VNF presence (binary)
        """
        # Chain sequence (padded)
        chain_seq = np.full(max_length, -1, dtype=np.float32)
        for i, vnf in enumerate(chain[:max_length]):
            chain_seq[i] = vnf / config.NUM_VNF_TYPES  # Normalize
        
        # VNF presence
        vnf_presence = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        for vnf in chain:
            if vnf < config.NUM_VNF_TYPES:
                vnf_presence[vnf] = 1.0
        
        return np.concatenate([chain_seq, vnf_presence])
    
    @staticmethod
    def _aggregate_chain_stats(active_reqs, max_top_chains=5):
        """
        Aggregate statistics for top-K most common chain patterns.
        
        Returns:
            Array of shape (max_top_chains × features,) where features include:
            - Chain pattern encoding
            - Count (normalized)
            - Avg bandwidth
            - Avg remaining time
        """
        from collections import Counter
        
        # Count chain patterns
        chain_counter = Counter()
        chain_info = {}
        
        for req in active_reqs:
            chain_tuple = tuple(req.chain)
            chain_counter[chain_tuple] += 1
            
            if chain_tuple not in chain_info:
                chain_info[chain_tuple] = {
                    'bw_sum': 0.0,
                    'rem_time_sum': 0.0,
                    'count': 0
                }
            
            chain_info[chain_tuple]['bw_sum'] += req.bandwidth
            chain_info[chain_tuple]['rem_time_sum'] += req.get_remaining_time()
            chain_info[chain_tuple]['count'] += 1
        
        # Get top-K chains
        top_chains = chain_counter.most_common(max_top_chains)
        
        result = []
        for chain_tuple, count in top_chains:
            # Chain pattern
            pattern = Observer._encode_chain_pattern(list(chain_tuple))
            
            # Stats
            info = chain_info[chain_tuple]
            stats = np.array([
                count / max(1, len(active_reqs)),  # Normalized count
                info['bw_sum'] / count / 1000.0,   # Avg BW
                info['rem_time_sum'] / count / 100.0  # Avg remaining time
            ], dtype=np.float32)
            
            result.append(np.concatenate([pattern, stats]))
        
        # Pad to max_top_chains
        feature_size = 4 + config.NUM_VNF_TYPES + 3  # max_length + NUM_VNF + 3 stats
        while len(result) < max_top_chains:
            result.append(np.zeros(feature_size, dtype=np.float32))
        
        return np.concatenate(result)
    
    @staticmethod
    def get_drl_observation(dc, sfc_manager, active_reqs=None):
        """
        Enhanced DRL observation with chain pattern encoding.
        
        Architecture:
        - s1: DC state (resources + VNF installation)
        - s2: DC-specific demand (VNF counts + chain patterns at this DC)
        - s3: Global demand (top-K chain patterns + aggregated stats)
        """
        if active_reqs is None:
            active_reqs = Observer.get_active_requests(sfc_manager)

        is_server = getattr(dc, 'is_server', True)
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}

        # ============== S1: DC State ==============
        cpu_val = getattr(dc, 'cpu', 0.0) if getattr(dc, 'cpu', 0.0) is not None else 0.0
        ram_val = getattr(dc, 'ram', 0.0) if getattr(dc, 'ram', 0.0) is not None else 0.0

        res_state = np.array([
            cpu_val / config.DC_CPU_CYCLES,
            ram_val / config.DC_RAM
        ], dtype=np.float32)
        
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        if is_server and hasattr(dc, 'installed_vnfs'):
            for vnf in dc.installed_vnfs:
                idx = vnf_map.get(vnf.vnf_type, 0)
                installed_counts[idx] += 1
                if vnf.is_idle(): 
                    idle_counts[idx] += 1
            
            installed_counts /= 10.0
            idle_counts /= 10.0
        
        s1 = np.concatenate([res_state, installed_counts, idle_counts], dtype=np.float32)

        # ============== S2: DC-Specific Demand ==============
        # VNF demand counts
        vnf_demand_at_dc = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        # Chain pattern demand (top-3 at this DC)
        dc_chains = []
        for req in active_reqs:
            if req.source == dc.id:
                for vnf_type in req.chain:
                    idx = vnf_map.get(vnf_type, vnf_type) if isinstance(vnf_type, int) else 0
                    if idx < config.NUM_VNF_TYPES:
                        vnf_demand_at_dc[idx] += 1
                dc_chains.append(req)
        
        vnf_demand_at_dc /= max(1, len(active_reqs))
        
        # Encode top-3 chain patterns at this DC
        chain_patterns_dc = Observer._aggregate_chain_stats(dc_chains, max_top_chains=3)
        
        s2 = np.concatenate([vnf_demand_at_dc, chain_patterns_dc], dtype=np.float32)

        # ============== S3: Global Demand ==============
        total_count = len(active_reqs) / 50.0
        avg_rem = np.mean([r.get_remaining_time() for r in active_reqs]) / 100.0 if active_reqs else 1.0
        avg_bw = np.mean([r.bandwidth for r in active_reqs]) / 1000.0 if active_reqs else 0.0
        
        total_finished = len(sfc_manager.completed_history) + len(sfc_manager.dropped_history)
        dropped_rate = len(sfc_manager.dropped_history) / total_finished if total_finished > 0 else 0.0
        
        # Global VNF demand
        global_vnf_demand = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        if active_reqs:
            for req in active_reqs:
                for vnf_type in req.chain:
                    idx = vnf_map.get(vnf_type, vnf_type) if isinstance(vnf_type, int) else 0
                    if idx < config.NUM_VNF_TYPES:
                        global_vnf_demand[idx] += 1
            global_vnf_demand /= max(1, len(active_reqs))
        
        # Top-5 global chain patterns
        chain_patterns_global = Observer._aggregate_chain_stats(active_reqs, max_top_chains=5)
        
        s3 = np.concatenate([
            np.array([total_count, avg_rem, avg_bw, dropped_rate], dtype=np.float32),
            global_vnf_demand,
            chain_patterns_global
        ], dtype=np.float32)

        return (s1, s2, s3)