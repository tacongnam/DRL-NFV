import numpy as np
import config

def calculate_priority(req, dc_id):
    # --- P1: Time Elapsed vs Delay Constraint ---
    # Formula: P1 = TE^s - D^s
    # Ý nghĩa: Giá trị âm, tăng dần về 0 khi thời gian trôi qua.
    p1 = req.elapsed_time - req.max_delay

    # --- P2: SFC Chain Location Affinity ---
    # Bài báo: "If any of the VNFs... installed in current DC... high priority"
    # "degrade that VNF's priority" if allocated to other DCs.
    p2 = 0.0
    if len(req.placed_vnfs) > 0:
        # Lấy VNF vừa được đặt trước đó (Predecessor)
        _, last_dc_id = req.placed_vnfs[-1]
        
        if last_dc_id == dc_id:
            # Nếu VNF trước đó nằm ở DC này -> Ưu tiên cao để giảm delay truyền dẫn
            p2 = config.PRIORITY_P2_SAME_DC
        else:
            # Nếu VNF trước đó nằm ở DC khác -> Giảm ưu tiên
            p2 = config.PRIORITY_P2_DIFF_DC
    else:
        # Nếu đây là VNF đầu tiên của chuỗi
        # Kiểm tra xem DC này có gần Source của Request không (Optional logic)
        # Trong bài báo chỉ nói về "previous VNFs", nên ta để 0.
        p2 = 0.0

    # --- P3: Urgency ---
    # Formula: P3 = C / (D^s - TE^s + epsilon)
    # Chỉ tính khi thời gian còn lại nhỏ hơn ngưỡng Thr
    remaining_time = req.max_delay - req.elapsed_time
    p3 = 0.0
    
    # Nếu thời gian còn lại ít hơn ngưỡng quy định
    if remaining_time < config.URGENCY_THRESHOLD:
        # Tránh chia cho 0 bằng EPSILON (hoặc số âm nếu đã quá hạn)
        denominator = max(remaining_time, config.EPSILON) 
        p3 = config.URGENCY_CONSTANT_C / denominator
    
    # Tổng hợp
    priority = p1 + p2 + p3
    return priority

def get_valid_actions_mask(curr_dc, active_requests):
    """
    Generate action mask to prevent invalid actions
    Returns boolean array where True = valid action
    """
    mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
    
    # Action 0: WAIT - always valid
    mask[0] = True
    
    # Actions 1 to NUM_VNF_TYPES: UNINSTALL
    for i in range(config.NUM_VNF_TYPES):
        vnf_type = config.VNF_TYPES[i]
        
        # Check if we have idle VNF of this type to uninstall
        has_idle = False
        for v in curr_dc.installed_vnfs:
            if v.vnf_type == vnf_type and v.is_idle():
                has_idle = True
                break
        
        # Only allow uninstall if: (1) has idle instance AND (2) not critically needed
        if has_idle:
            # Check if any active request urgently needs this VNF
            is_urgent = False
            for req in active_requests:
                if not req.is_completed and not req.is_dropped:
                    next_vnf = req.get_next_vnf()
                    if next_vnf == vnf_type:
                        # Check urgency: remaining time < threshold
                        remaining = req.max_delay - req.elapsed_time
                        if remaining < config.URGENCY_THRESHOLD:
                            is_urgent = True
                            break
            
            if not is_urgent:
                mask[i + 1] = True
    
    # Actions (NUM_VNF_TYPES+1) to end: ALLOC
    for i in range(config.NUM_VNF_TYPES):
        vnf_type = config.VNF_TYPES[i]
        action_idx = config.NUM_VNF_TYPES + 1 + i
        
        # Check if: (1) has resources AND (2) some request needs this VNF
        if curr_dc.has_resources(vnf_type):
            has_demand = False
            for req in active_requests:
                if not req.is_completed and not req.is_dropped:
                    next_vnf = req.get_next_vnf()
                    if next_vnf == vnf_type:
                        has_demand = True
                        break
            
            if has_demand:
                mask[action_idx] = True
    
    # Fallback: If no action is valid (shouldn't happen), allow WAIT
    if not np.any(mask):
        mask[0] = True
    
    return mask

def get_dc_sfc_state_info(curr_dc, active_requests):
    """
    State 2: DC-SFC Info - shows SFC processing stages at current DC
    Matrix [|S| x (1 + 2*|V|)]
    """
    s2 = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES))
    
    for i, s_type in enumerate(config.SFC_TYPES):
        # Get requests of this type that involve current DC
        relevant_reqs = []
        for req in active_requests:
            if req.type == s_type and not req.is_completed and not req.is_dropped:
                # Check if this DC is involved in the request
                dc_involved = False
                
                # Check if any VNF already placed at this DC
                for vnf_name, dc_id in req.placed_vnfs:
                    if dc_id == curr_dc.id:
                        dc_involved = True
                        break
                
                # Check if next VNF could be placed here
                next_vnf = req.get_next_vnf()
                if next_vnf and curr_dc.has_resources(next_vnf):
                    dc_involved = True
                
                if dc_involved:
                    relevant_reqs.append(req)
        
        if relevant_reqs:
            # Set SFC type indicator
            s2[i, 0] = len(relevant_reqs)  # Number of this SFC type being processed
            
            # Count allocated and remaining VNFs
            allocated_counts = np.zeros(config.NUM_VNF_TYPES)
            remaining_counts = np.zeros(config.NUM_VNF_TYPES)
            
            for req in relevant_reqs:
                # Already allocated VNFs at this DC
                for vnf_name, dc_id in req.placed_vnfs:
                    if dc_id == curr_dc.id:
                        v_idx = config.VNF_TYPES.index(vnf_name)
                        allocated_counts[v_idx] += 1
                
                # Remaining VNFs in the chain
                for j in range(req.current_vnf_index, len(req.chain)):
                    vnf_name = req.chain[j]
                    v_idx = config.VNF_TYPES.index(vnf_name)
                    remaining_counts[v_idx] += 1
            
            s2[i, 1:config.NUM_VNF_TYPES+1] = allocated_counts
            s2[i, config.NUM_VNF_TYPES+1:] = remaining_counts
            
    return s2.flatten().astype(np.float32)