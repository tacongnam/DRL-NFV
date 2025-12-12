# environment/utils.py
import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests):
    """
    Tạo action mask để chặn invalid actions
    
    Args:
        curr_dc: DataCenter hiện tại
        active_requests: Danh sách active requests
        
    Returns:
        np.array: Boolean mask [ACTION_SPACE_SIZE]
    """
    mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
    
    # Action 0: WAIT - luôn valid
    mask[0] = True
    
    # Actions 1 -> NUM_VNF_TYPES: UNINSTALL
    for i, vnf_type in enumerate(config.VNF_TYPES):
        action_idx = i + 1
        
        # Valid nếu có idle VNF của loại này
        if curr_dc.count_idle_vnf_type(vnf_type) > 0:
            mask[action_idx] = True
    
    # Actions NUM_VNF_TYPES+1 -> 2*NUM_VNF_TYPES+1: ALLOCATION
    # Collect VNF types needed by active requests
    needed_vnfs = set()
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            next_vnf = req.get_next_vnf()
            if next_vnf:
                needed_vnfs.add(next_vnf)
    
    for i, vnf_type in enumerate(config.VNF_TYPES):
        action_idx = config.NUM_VNF_TYPES + 1 + i
        
        # Valid nếu:
        # 1. Có request cần VNF này
        # 2. DC có đủ tài nguyên HOẶC có idle VNF sẵn
        if vnf_type in needed_vnfs:
            has_idle = curr_dc.get_idle_vnf(vnf_type) is not None
            has_resources = curr_dc.has_resources(vnf_type)
            
            if has_idle or has_resources:
                mask[action_idx] = True
    
    return mask

def get_vnf_type_from_action(action):
    """
    Map action ID sang VNF type
    
    Returns:
        str hoặc None
    """
    if action == 0:
        return None  # WAIT
    
    vnf_idx = (action - 1) % config.NUM_VNF_TYPES
    return config.VNF_TYPES[vnf_idx]

def get_action_type(action):
    """
    Xác định loại action
    
    Returns:
        str: 'wait', 'uninstall', hoặc 'allocate'
    """
    if action == 0:
        return 'wait'
    elif 1 <= action <= config.NUM_VNF_TYPES:
        return 'uninstall'
    else:
        return 'allocate'