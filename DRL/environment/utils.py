import numpy as np
import config

def get_vnf_name_from_action(action):
    """Giải mã tên VNF từ action ID"""
    if action == 0:
        return None  # WAIT
    elif 1 <= action <= config.NUM_VNF_TYPES:
        return config.VNF_TYPES[action - 1]  # UNINSTALL
    else:
        return config.VNF_TYPES[action - (config.NUM_VNF_TYPES + 1)]  # ALLOC

def calculate_priority(req, dc_id):
    """Tính độ ưu tiên Request"""
    # P1: Time
    p1 = req.elapsed_time - req.max_delay
    
    # P2: Affinity
    p2 = 0.0
    if len(req.placed_vnfs) > 0:
        _, last_dc_id = req.placed_vnfs[-1]
        if last_dc_id == dc_id:
            p2 = config.PRIORITY_P2_SAME_DC
        else:
            p2 = config.PRIORITY_P2_DIFF_DC

    # P3: Urgency
    remaining_time = req.max_delay - req.elapsed_time
    p3 = 0.0
    if remaining_time < config.URGENCY_THRESHOLD:
        p3 = config.URGENCY_CONSTANT_C / (max(remaining_time, config.EPSILON))
    
    return p1 + p2 + p3

def get_valid_actions_mask(curr_dc, active_requests):
    """
    Masking: Chặn các hành động không hợp lệ.
    Tra ve: np.array([True, False, ...])
    """
    mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
    mask[0] = True  # WAIT luôn luôn được phép
    
    # 1. Check Uninstall (Actions 1 -> N)
    # Chỉ được Uninstall nếu DC đang có VNF đó và nó đang rảnh
    for i in range(config.NUM_VNF_TYPES):
        vnf_type = config.VNF_TYPES[i]
        # Tìm xem có VNF loại này đang idle không
        has_idle = any(v.vnf_type == vnf_type and v.is_idle() for v in curr_dc.installed_vnfs)
        if has_idle:
            mask[i + 1] = True
            
    # 2. Check Allocation (Actions N+1 -> 2N+1)
    # Chỉ được Alloc nếu DC đủ tài nguyên VÀ có Request đang cần loại VNF đó
    req_needs = set()
    for r in active_requests:
        if not r.is_completed and not r.is_dropped:
            vnf = r.get_next_vnf()
            if vnf: req_needs.add(vnf)

    for i in range(config.NUM_VNF_TYPES):
        vnf_type = config.VNF_TYPES[i]
        action_idx = config.NUM_VNF_TYPES + 1 + i
        
        # Điều kiện 1: Có request cần
        if vnf_type in req_needs:
            # Điều kiện 2: Đủ tài nguyên
            if curr_dc.has_resources(vnf_type):
                mask[action_idx] = True
    
    return mask