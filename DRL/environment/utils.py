import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests):
    """
    Create action mask to block invalid actions
    """
    mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
    
    # Action 0: WAIT - always valid
    mask[0] = True
    
    # Actions 1 -> NUM_VNF_TYPES: UNINSTALL
    # Valid if there's an idle VNF of this type
    # Count idle VNFs directly from installed_vnfs
    idle_counts = {}
    for vnf in curr_dc.installed_vnfs:
        if vnf.is_idle():
            idle_counts[vnf.vnf_type] = idle_counts.get(vnf.vnf_type, 0) + 1

    for i, vnf_type in enumerate(config.VNF_TYPES):
        action_idx = i + 1
        if idle_counts.get(vnf_type, 0) > 0:
            mask[action_idx] = True
    
    # Actions NUM_VNF_TYPES+1 -> 2*NUM_VNF_TYPES: ALLOCATION
    # Collect VNF types needed by active requests
    needed_vnfs = set()
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            next_vnf = req.get_next_vnf()
            if next_vnf is not None:
                needed_vnfs.add(next_vnf)
    
    for i, vnf_type in enumerate(config.VNF_TYPES):
        action_idx = config.NUM_VNF_TYPES + 1 + i
        
        if vnf_type in needed_vnfs:
            # Check availability: Has Idle VNF OR Has Resources to install new
            has_idle = idle_counts.get(vnf_type, 0) > 0
            has_resources = curr_dc.has_resources(vnf_type)
            
            if has_idle or has_resources:
                mask[action_idx] = True
    
    return mask

def get_vnf_type_from_action(action):
    if action == 0:
        return None
    vnf_idx = (action - 1) % config.NUM_VNF_TYPES
    return config.VNF_TYPES[vnf_idx]

def get_action_type(action):
    if action == 0:
        return 'wait'
    elif 1 <= action <= config.NUM_VNF_TYPES:
        return 'uninstall'
    else:
        return 'allocate'