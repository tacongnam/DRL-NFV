import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests):
    """Enhanced masking with deadline-aware constraints"""
    action_space_size = 1 + 2 * config.MAX_VNF_TYPES
    mask = np.zeros(action_space_size, dtype=bool)
    
    mask[0] = True
    
    # Count idle and installed VNFs
    idle_counts = {}
    installed_counts = {}
    for vnf in curr_dc.installed_vnfs:
        vnf_type = vnf.vnf_type
        installed_counts[vnf_type] = installed_counts.get(vnf_type, 0) + 1
        if vnf.is_idle():
            idle_counts[vnf_type] = idle_counts.get(vnf_type, 0) + 1

    # Analyze needed VNFs with urgency
    needed_vnfs = {}
    urgent_vnfs = set()
    near_expiry_vnfs = set()
    
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            remaining = req.get_remaining_time()
            
            # Skip requests that are about to expire anyway
            if remaining < 1.0:
                continue
            
            next_vnf = req.get_next_vnf()
            if next_vnf is not None and next_vnf < config.MAX_VNF_TYPES:
                needed_vnfs[next_vnf] = needed_vnfs.get(next_vnf, 0) + 1
                
                if remaining < config.URGENCY_THRESHOLD:
                    urgent_vnfs.add(next_vnf)
                if remaining < 10.0:
                    near_expiry_vnfs.add(next_vnf)
    
    # UNINSTALL actions: conservative approach
    for vnf_type in range(config.NUM_VNF_TYPES):
        if vnf_type >= config.MAX_VNF_TYPES:
            continue
            
        action_idx = vnf_type + 1
        idle_count = idle_counts.get(vnf_type, 0)
        needed_count = needed_vnfs.get(vnf_type, 0)
        
        # Only allow uninstall if:
        # 1. Have idle VNFs
        # 2. Idle count significantly exceeds need
        # 3. Not urgent or near expiry
        if idle_count > 0:
            if vnf_type in urgent_vnfs:
                # Never uninstall urgent VNFs unless we have many extras
                if idle_count > needed_count + 2:
                    mask[action_idx] = True
            elif vnf_type in near_expiry_vnfs:
                # Be cautious with near-expiry VNFs
                if idle_count > needed_count + 1:
                    mask[action_idx] = True
            else:
                # Normal case: allow if idle > needed
                if idle_count > needed_count:
                    mask[action_idx] = True
    
    # ALLOCATE actions: prioritize urgent and feasible
    for vnf_type in range(config.NUM_VNF_TYPES):
        if vnf_type >= config.MAX_VNF_TYPES:
            continue
            
        action_idx = config.MAX_VNF_TYPES + 1 + vnf_type
        
        if vnf_type in needed_vnfs:
            has_idle = idle_counts.get(vnf_type, 0) > 0
            has_resources = curr_dc.has_resources(vnf_type)
            
            # Always allow if idle VNF exists
            if has_idle:
                mask[action_idx] = True
            # Allow install if:
            # 1. Resources available
            # 2. VNF is needed (urgent priority)
            elif has_resources:
                # Prioritize urgent and near-expiry
                if vnf_type in urgent_vnfs or vnf_type in near_expiry_vnfs:
                    mask[action_idx] = True
                # Allow installation if we don't have this VNF at all
                elif installed_counts.get(vnf_type, 0) == 0:
                    mask[action_idx] = True
                # Allow if demand is high
                elif needed_vnfs[vnf_type] >= 2:
                    mask[action_idx] = True
    
    # Ensure at least one valid action
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