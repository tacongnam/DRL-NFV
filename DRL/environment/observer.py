# environment/observer.py
import numpy as np
import config

class Observer:
    """Tạo state representation cho DRL model"""
    
    @staticmethod
    def get_full_state(curr_dc, sfc_manager):
        """
        Tạo state gồm 3 input layers
        
        Returns:
            tuple: (s1, s2, s3)
            - s1: DC state [1 x (2|V|+2)]
            - s2: DC-SFC state [|S| x (1 + 2|V|)] -> flatten
            - s3: Global SFC state [|S| x (4 + |V|)] -> flatten
        """
        # --- INPUT 1: DC State ---
        # [CPU available, Storage available, Installed VNFs count per type, Idle VNFs count per type]
        s1 = np.array([curr_dc.cpu, curr_dc.storage], dtype=np.float32)
        
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        
        for vnf in curr_dc.installed_vnfs:
            idx = vnf_map[vnf.vnf_type]
            installed_counts[idx] += 1
            if vnf.is_idle():
                idle_counts[idx] += 1
        
        s1 = np.concatenate([s1, installed_counts, idle_counts])
        
        # --- INPUT 2 & 3: Tối ưu bằng cách nhóm request theo type ---
        reqs_by_type = {t: [] for t in config.SFC_TYPES}
        for req in sfc_manager.active_requests:
            if not req.is_completed and not req.is_dropped:
                reqs_by_type[req.type].append(req)
        
        s2_matrix = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES), dtype=np.float32)
        s3_matrix = np.zeros((config.NUM_SFC_TYPES, 4 + config.NUM_VNF_TYPES), dtype=np.float32)
        
        for i, sfc_type in enumerate(config.SFC_TYPES):
            reqs = reqs_by_type[sfc_type]
            
            # --- INPUT 3: Global State ---
            # [Count, Avg remaining time, BW requirement, Total pending VNFs, Pending VNF counts per type]
            count = len(reqs)
            s3_matrix[i, 0] = count
            
            if count > 0:
                avg_remaining = np.mean([r.get_remaining_time() for r in reqs])
                s3_matrix[i, 1] = avg_remaining
            
            s3_matrix[i, 2] = config.SFC_SPECS[sfc_type]['bw']
            
            pending_vnfs = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            total_pending = 0
            
            # --- INPUT 2: DC-specific State ---
            # [DC-relevant count, Allocated VNFs in this DC per type, Remaining VNFs per type]
            dc_relevant_count = 0
            allocated_in_dc = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            remaining_in_chain = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            
            for req in reqs:
                # Global: pending VNFs
                next_vnf = req.get_next_vnf()
                if next_vnf:
                    pending_vnfs[vnf_map[next_vnf]] += 1
                    total_pending += 1
                
                # DC-specific: check if DC is involved
                is_involved = False
                
                # Check if any VNF was placed in this DC
                for vnf_name, dc_id, _, _ in req.placed_vnfs:
                    if dc_id == curr_dc.id:
                        allocated_in_dc[vnf_map[vnf_name]] += 1
                        is_involved = True
                
                # Check if next VNF can be placed here
                if next_vnf and curr_dc.has_resources(next_vnf):
                    is_involved = True
                
                if is_involved:
                    dc_relevant_count += 1
                    # Count remaining VNFs in chain
                    for k in range(req.current_vnf_index, len(req.chain)):
                        remaining_in_chain[vnf_map[req.chain[k]]] += 1
            
            # Fill matrices
            s3_matrix[i, 3] = total_pending
            s3_matrix[i, 4:] = pending_vnfs
            
            s2_matrix[i, 0] = dc_relevant_count
            s2_matrix[i, 1:1+config.NUM_VNF_TYPES] = allocated_in_dc
            s2_matrix[i, 1+config.NUM_VNF_TYPES:] = remaining_in_chain
        
        return (s1, s2_matrix.flatten(), s3_matrix.flatten())