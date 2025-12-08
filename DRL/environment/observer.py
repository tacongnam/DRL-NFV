import numpy as np
import config

class Observer:
    @staticmethod
    def get_full_state(curr_dc, sfc_manager):
        # --- Input 1: DC State [1 x (2|V|+2)] ---
        s1 = np.array([curr_dc.cpu, curr_dc.storage], dtype=np.float32)
        installed = np.zeros(config.NUM_VNF_TYPES)
        idle = np.zeros(config.NUM_VNF_TYPES)
        
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        
        for v in curr_dc.installed_vnfs:
            idx = vnf_map[v.vnf_type]
            installed[idx] += 1
            if v.is_idle():
                idle[idx] += 1
        s1 = np.concatenate([s1, installed, idle])

        # --- Input 2 & 3: Optimized Loop ---
        # Pre-calculate active reqs per type
        reqs_by_type = {t: [] for t in config.SFC_TYPES}
        for r in sfc_manager.active_requests:
            if not r.is_completed and not r.is_dropped:
                reqs_by_type[r.type].append(r)

        s2_matrix = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES))
        s3_matrix = np.zeros((config.NUM_SFC_TYPES, 4 + config.NUM_VNF_TYPES))

        for i, s_type in enumerate(config.SFC_TYPES):
            reqs = reqs_by_type[s_type]
            
            # -- State 3 Global --
            count = len(reqs)
            s3_matrix[i, 0] = count
            if count > 0:
                avg_rem_time = np.mean([r.max_delay - r.elapsed_time for r in reqs])
                s3_matrix[i, 1] = avg_rem_time
            s3_matrix[i, 2] = config.SFC_SPECS[s_type]['bw']
            
            pending_vnfs_global = np.zeros(config.NUM_VNF_TYPES)
            total_pending = 0
            
            # -- State 2 DC Specific --
            dc_relevant_count = 0
            allocated_dc = np.zeros(config.NUM_VNF_TYPES)
            remaining_chain = np.zeros(config.NUM_VNF_TYPES)

            for r in reqs:
                # Global Pending stats
                next_v = r.get_next_vnf()
                if next_v:
                    pending_vnfs_global[vnf_map[next_v]] += 1
                    total_pending += 1
                
                # DC Specific stats
                # Check if DC is involved (has placed VNF or can place next)
                involved = False
                for vn, did in r.placed_vnfs:
                    if did == curr_dc.id:
                        allocated_dc[vnf_map[vn]] += 1
                        involved = True
                
                if next_v and curr_dc.has_resources(next_v):
                    involved = True
                
                if involved:
                    dc_relevant_count += 1
                    # Remaining in chain
                    for k in range(r.current_vnf_index, len(r.chain)):
                        remaining_chain[vnf_map[r.chain[k]]] += 1

            s3_matrix[i, 3] = total_pending
            s3_matrix[i, 4:] = pending_vnfs_global
            
            s2_matrix[i, 0] = dc_relevant_count
            s2_matrix[i, 1 : 1+config.NUM_VNF_TYPES] = allocated_dc
            s2_matrix[i, 1+config.NUM_VNF_TYPES :] = remaining_chain

        return (s1, s2_matrix.flatten(), s3_matrix.flatten())