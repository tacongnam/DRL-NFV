import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests, topology):
    """
    Tạo mask các action hợp lệ.
    """
    action_space_size = config.ACTION_SPACE_SIZE
    mask = np.zeros(action_space_size, dtype=bool)
    mask[0] = True # Action 0: WAIT luôn luôn hợp lệ
    
    # 1. Thống kê VNF đang rảnh tại DC hiện tại
    idle_counts = {}
    for vnf in curr_dc.installed_vnfs:
        if vnf.is_idle():
            idle_counts[vnf.vnf_type] = idle_counts.get(vnf.vnf_type, 0) + 1

    # 2. Tìm Request ưu tiên nhất (Target Request)
    # Trong Easy mode, việc xác định đúng request để phục vụ là quan trọng nhất
    target_req = None
    min_deadline = float('inf')
    needed_vnfs = set() # Tổng hợp nhu cầu của TẤT CẢ request

    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            # Lấy VNF tiếp theo cần của request này
            nxt = req.get_next_vnf()
            if nxt is not None and nxt < config.MAX_VNF_TYPES:
                needed_vnfs.add(nxt)
                
                # Logic chọn Target Request: Ưu tiên cái sắp hết hạn hoặc đang xử lý dở
                rem_time = req.get_remaining_time()
                # Ưu tiên cực cao cho request đang ở tại node này (cần đi tiếp)
                if req.get_last_placed_dc() == curr_dc.id:
                    rem_time -= 1000 # Boost priority
                
                if rem_time < min_deadline:
                    min_deadline = rem_time
                    target_req = req

    # 3. Action UNINSTALL
    for vnf_type in range(config.NUM_VNF_TYPES):
        # Chỉ cho phép gỡ nếu có dư VNF rảnh
        if idle_counts.get(vnf_type, 0) > 0:
             mask[vnf_type + 1] = True

    # 4. Action ALLOCATE
    # Logic: Chỉ bật mask ALLOCATE nếu thỏa mãn Connectivity cho Target Request
    for vnf_type in range(config.NUM_VNF_TYPES):
        if vnf_type in needed_vnfs:
            action_idx = config.MAX_VNF_TYPES + 1 + vnf_type
            
            # Check Resource trước (nhanh)
            can_host = (idle_counts.get(vnf_type, 0) > 0) or curr_dc.has_resources(vnf_type)
            
            if can_host:
                # --- CRITICAL FIX FOR EASY MODE ---
                # Nếu loại VNF này khớp với Target Request, ta PHẢI kiểm tra kết nối.
                # Nếu không kiểm tra, Agent sẽ đặt bừa vào node cụt -> Fail 100% ở Easy mode.
                is_connected = True
                if target_req and target_req.get_next_vnf() == vnf_type:
                    last_node = target_req.get_last_placed_dc()
                    if last_node != curr_dc.id:
                        # Dùng hàm check nhanh mới viết trong topology
                        is_connected = topology.check_connectivity(last_node, curr_dc.id, target_req.bandwidth)
                
                if is_connected:
                    mask[action_idx] = True

    return mask

def get_vnf_type_from_action(action):
    if action == 0: return None
    if action <= config.MAX_VNF_TYPES:
        vnf_idx = action - 1
    else:
        vnf_idx = action - (config.MAX_VNF_TYPES + 1)
        
    if 0 <= vnf_idx < config.NUM_VNF_TYPES:
        return vnf_idx
    return None

def get_action_type(action):
    if action == 0: return 'wait'
    elif 1 <= action <= config.MAX_VNF_TYPES: return 'uninstall'
    else: return 'allocate'