import numpy as np
import config

class DCStateObserver:
    """Extract DC state cho GenAI model"""
    
    @staticmethod
    def get_dc_state(dc, sfc_manager):
        """
        Trích xuất trạng thái của 1 DC
        
        Features:
        - Resources: CPU, RAM, Storage (normalized)
        - VNF counts: Installed, Idle per type
        - SFC info: Source count, min E2E remaining, BW needs
        
        Returns:
            np.array shape (state_dim,)
        """
        # 1. Resources (normalized)
        cpu_ratio = dc.cpu / config.DC_CPU_CYCLES
        ram_ratio = dc.ram / config.DC_RAM
        storage_ratio = dc.storage / config.DC_STORAGE
        
        # 2. VNF counts
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        for vnf in dc.installed_vnfs:
            idx = vnf_map[vnf.vnf_type]
            installed_counts[idx] += 1
            if vnf.is_idle():
                idle_counts[idx] += 1
        
        # Normalize by max possible (arbitrary limit: 10)
        installed_counts = installed_counts / 10.0
        idle_counts = idle_counts / 10.0
        
        # 3. SFC-related info
        sfc_source_count = 0  # Số SFC có source tại DC này
        min_remaining_time = 999.0  # ms
        total_bw_need = 0.0  # Mbps
        
        active_reqs = [r for r in sfc_manager.active_requests 
                       if not r.is_completed and not r.is_dropped]
        
        for req in active_reqs:
            if req.source == dc.id:
                sfc_source_count += 1
            
            remaining = req.get_remaining_time()
            if remaining < min_remaining_time:
                min_remaining_time = remaining
            
            # Check if DC is relevant
            next_vnf = req.get_next_vnf()
            if next_vnf:
                has_vnf = dc.get_idle_vnf(next_vnf) is not None
                has_res = dc.has_resources(next_vnf)
                if has_vnf or has_res:
                    total_bw_need += req.specs['bw']
        
        # Normalize
        sfc_source_count = min(sfc_source_count / 10.0, 1.0)
        min_remaining_time = min_remaining_time / 100.0  # Scale to ~0-1
        total_bw_need = total_bw_need / 1000.0  # Normalize by 1 Gbps
        
        # Combine all features
        state = np.concatenate([
            [cpu_ratio, ram_ratio, storage_ratio],
            installed_counts,
            idle_counts,
            [sfc_source_count, min_remaining_time, total_bw_need]
        ], dtype=np.float32)
        
        return state
    
    @staticmethod
    def get_all_dc_states(dcs, sfc_manager):
        """
        Lấy state của tất cả DC
        
        Returns:
            np.array shape (num_dcs, state_dim)
        """
        states = []
        for dc in dcs:
            state = DCStateObserver.get_dc_state(dc, sfc_manager)
            states.append(state)
        return np.array(states, dtype=np.float32)
    
    @staticmethod
    def get_state_dim():
        """Tính state dimension"""
        # 3 (resources) + 2*6 (VNF counts) + 3 (SFC info)
        return 3 + 2 * config.NUM_VNF_TYPES + 3
    
    @staticmethod
    def calculate_dc_value(dc, sfc_manager, dc_prev_state):
        """
        Tính value cho DC (để train Value Network)
        
        Value dựa trên:
        - Số SFC có thể satisfy từ DC này
        - Resource efficiency
        - Proximity to urgent requests
        
        Returns:
            float: value score
        """
        value = 0.0
        
        active_reqs = [r for r in sfc_manager.active_requests 
                       if not r.is_completed and not r.is_dropped]
        
        # Factor 1: Capable of serving requests
        for req in active_reqs:
            next_vnf = req.get_next_vnf()
            if next_vnf:
                has_idle = dc.get_idle_vnf(next_vnf) is not None
                has_res = dc.has_resources(next_vnf)
                
                if has_idle or has_res:
                    # Urgent requests get higher value
                    remaining = req.get_remaining_time()
                    urgency_factor = 1.0 / (remaining + 1.0)
                    value += 10.0 * urgency_factor
        
        # Factor 2: Resource availability (encourage using free DCs)
        cpu_avail = dc.cpu / config.DC_CPU_CYCLES
        value += cpu_avail * 5.0
        
        # Factor 3: Source DC bonus
        source_count = sum(1 for r in active_reqs if r.source == dc.id)
        value += source_count * 15.0
        
        return value