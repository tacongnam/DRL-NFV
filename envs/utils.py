import numpy as np
import config

def get_valid_actions_mask(curr_dc, active_requests, topology):
    action_space_size = 1 + 2 * config.MAX_VNF_TYPES
    mask = np.zeros(action_space_size, dtype=bool)
    mask[0] = True # WAIT
    
    # 1. Thống kê VNF tại DC
    idle_counts = {}
    for vnf in curr_dc.installed_vnfs:
        if vnf.is_idle():
            idle_counts[vnf.vnf_type] = idle_counts.get(vnf.vnf_type, 0) + 1

    # 2. Demand & Connectivity
    needed_vnfs = {}
    target_req = None
    min_deadline = float('inf')

    # Tìm request ưu tiên nhất
    for req in active_requests:
        if not req.is_completed and not req.is_dropped:
            rem_time = req.get_remaining_time()
            if rem_time < min_deadline and rem_time > 0.5:
                min_deadline = rem_time
                target_req = req
            
            nxt = req.get_next_vnf()
            if nxt is not None and nxt < config.MAX_VNF_TYPES:
                needed_vnfs[nxt] = needed_vnfs.get(nxt, 0) + 1

    # 3. Action UNINSTALL
    for vnf_type in range(config.NUM_VNF_TYPES):
        if idle_counts.get(vnf_type, 0) > needed_vnfs.get(vnf_type, 0):
             mask[vnf_type + 1] = True

    # 4. Connectivity Check (CẬP NHẬT: Kiểm tra Băng thông)
    network_ok = True
    if target_req:
        last_node = target_req.get_last_placed_dc()
        if last_node != curr_dc.id:
            # Dùng hàm check_connectivity mới trong topology.py
            network_ok = topology.check_connectivity(last_node, curr_dc.id, target_req.bandwidth)

    # 5. Action ALLOCATE
    if network_ok:
        for vnf_type in range(config.NUM_VNF_TYPES):
            action_idx = config.MAX_VNF_TYPES + 1 + vnf_type
            has_idle = idle_counts.get(vnf_type, 0) > 0
            
            if has_idle:
                 if needed_vnfs.get(vnf_type, 0) > 0:
                     mask[action_idx] = True
            elif curr_dc.has_resources(vnf_type):
                if needed_vnfs.get(vnf_type, 0) > 0:
                    mask[action_idx] = True

    return mask

def get_vnf_type_from_action(action):
    if action == 0: return None
    vnf_idx = (action - 1) % config.MAX_VNF_TYPES
    if vnf_idx < config.NUM_VNF_TYPES: return vnf_idx
    return None

def get_action_type(action):
    if action == 0: return 'wait'
    elif 1 <= action <= config.MAX_VNF_TYPES: return 'uninstall'
    else: return 'allocate'