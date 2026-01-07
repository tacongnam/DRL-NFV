import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests):
    action_space_size = 1 + 2 * config.MAX_VNF_TYPES
    mask = np.zeros(action_space_size, dtype=bool)
    
    mask[0] = True
    
    idle_counts = {}
    for vnf in curr_dc.installed_vnfs:
        if vnf.is_idle():
            idle_counts[vnf.vnf_type] = idle_counts.get(vnf.vnf_type, 0) + 1

    needed_vnfs = {}
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            next_vnf = req.get_next_vnf()
            if next_vnf is not None and next_vnf < config.MAX_VNF_TYPES:
                needed_vnfs[next_vnf] = needed_vnfs.get(next_vnf, 0) + 1
    
    for vnf_type in range(config.NUM_VNF_TYPES):
        if vnf_type >= config.MAX_VNF_TYPES:
            continue
            
        action_idx = vnf_type + 1
        
        idle_count = idle_counts.get(vnf_type, 0)
        needed_count = needed_vnfs.get(vnf_type, 0)
        
        if idle_count > needed_count:
            mask[action_idx] = True
    
    for vnf_type in range(config.NUM_VNF_TYPES):
        if vnf_type >= config.MAX_VNF_TYPES:
            continue
            
        action_idx = config.MAX_VNF_TYPES + 1 + vnf_type
        
        if vnf_type in needed_vnfs:
            has_idle = idle_counts.get(vnf_type, 0) > 0
            has_resources = curr_dc.has_resources(vnf_type)
            
            if has_idle or has_resources:
                mask[action_idx] = True
    
    if not np.any(mask[1:]):
        mask[0] = True
    
    return mask

def get_vnf_type_from_action(action):
    if action == 0:
        return None
    vnf_idx = (action - 1) % config.MAX_VNF_TYPES
    if vnf_idx < config.NUM_VNF_TYPES:
        return vnf_idx
    return None

def get_action_type(action):
    if action == 0:
        return 'wait'
    elif 1 <= action <= config.MAX_VNF_TYPES:
        return 'uninstall'
    else:
        return 'allocate'