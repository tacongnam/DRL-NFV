import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests, topology):
    """
    Mask optimized: 
    1. Kiểm tra tài nguyên Node (CPU/RAM).
    2. Chỉ kiểm tra kết nối mạng (Topology) CHO REQUEST ƯU TIÊN NHẤT.
    """
    action_space_size = 1 + 2 * config.MAX_VNF_TYPES
    mask = np.zeros(action_space_size, dtype=bool)
    mask[0] = True # Always allow WAIT
    
    # --- 1. Thống kê VNF tại DC hiện tại ---
    idle_counts = {}
    installed_counts = {} # 0: không có, 1: có
    
    for vnf in curr_dc.installed_vnfs:
        vnf_type = vnf.vnf_type
        installed_counts[vnf_type] = 1 
        if vnf.is_idle():
            idle_counts[vnf_type] = idle_counts.get(vnf_type, 0) + 1

    # --- 2. Thống kê nhu cầu (Demand) ---
    needed_vnfs = {}
    target_req = None
    min_deadline = float('inf')

    # Lọc request và tìm request gấp nhất để check mạng
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            # Logic tìm request ưu tiên (deadline gần nhất)
            rem_time = req.get_remaining_time()
            if rem_time < min_deadline and rem_time > 0.5: # 0.5 là ngưỡng an toàn
                min_deadline = rem_time
                target_req = req

            nxt = req.get_next_vnf()
            if nxt is not None and nxt < config.MAX_VNF_TYPES:
                needed_vnfs[nxt] = needed_vnfs.get(nxt, 0) + 1

    # --- 3. Mở Action UNINSTALL ---
    for vnf_type in range(config.NUM_VNF_TYPES):
        action_idx = vnf_type + 1
        # Chỉ uninstall nếu số lượng rảnh > số lượng cần
        if idle_counts.get(vnf_type, 0) > needed_vnfs.get(vnf_type, 0):
             mask[action_idx] = True

    # --- 4. Logic mạng (Connectivity) ---
    network_ok = True
    if target_req:
        last_node = target_req.get_last_placed_dc()
        if last_node != curr_dc.id:
            # Gọi hàm check nhanh (BFS)
            network_ok = topology.check_connectivity(last_node, curr_dc.id, target_req.bandwidth)

    # --- 5. Mở Action ALLOCATE ---
    if network_ok:
        for vnf_type in range(config.NUM_VNF_TYPES):
            action_idx = config.MAX_VNF_TYPES + 1 + vnf_type
            
            # Điều kiện Allocate:
            # 1. Có sẵn VNF đang rảnh (Idle) -> Ưu tiên dùng lại
            # 2. Chưa có VNF nhưng Server đủ tài nguyên -> Cài mới
            
            has_idle = idle_counts.get(vnf_type, 0) > 0
            
            # Nếu có idle, luôn cho phép (không tốn tài nguyên cài đặt)
            if has_idle:
                 if needed_vnfs.get(vnf_type, 0) > 0:
                     mask[action_idx] = True
            
            # Nếu không có idle, kiểm tra tài nguyên server để cài mới
            elif curr_dc.has_resources(vnf_type):
                # Chỉ cài mới nếu thực sự cần VNF này
                if needed_vnfs.get(vnf_type, 0) > 0:
                    mask[action_idx] = True
                
                # (Optional) Exploration: Cho phép cài VNF chưa từng có tại DC này
                # Giúp Agent học cách chuẩn bị trước VNF
                elif installed_counts.get(vnf_type, 0) == 0:
                    # Chỉ mở exploration với xác suất nhỏ hoặc điều kiện lỏng hơn nếu cần
                    # Ở đây ta giữ chặt để tránh lãng phí tài nguyên
                    pass 

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