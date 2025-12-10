# environment/controller.py
import numpy as np
import config
from spaces.vnf import VNFInstance

class ActionController:
    """Controller xử lý các action từ DRL model"""
    
    def __init__(self, sfc_manager, topology_manager):
        self.manager = sfc_manager
        self.topology = topology_manager
        self.accepted_count = 0
        self.dropped_count = 0

    def execute_action(self, action, curr_dc):
        """
        Thực thi action
        
        Args:
            action: Action ID (0: Wait, 1-N: Uninstall, N+1-2N: Alloc)
            curr_dc: DataCenter hiện tại
            
        Returns:
            (reward, is_sfc_completed)
        """
        # Action 0: WAIT
        if action == 0:
            return config.REWARD_WAIT, False
        
        vnf_idx = (action - 1) % config.NUM_VNF_TYPES
        vnf_name = config.VNF_TYPES[vnf_idx]
        is_alloc = action > config.NUM_VNF_TYPES
        
        if is_alloc:
            return self._handle_allocation(curr_dc, vnf_name)
        else:
            return self._handle_uninstall(curr_dc, vnf_name)

    def _handle_allocation(self, dc, vnf_name):
        """Xử lý action Allocation"""
        # 1. Tìm request có priority cao nhất cần VNF này
        candidates = []
        for req in self.manager.active_requests:
            if not req.is_completed and not req.is_dropped and req.get_next_vnf() == vnf_name:
                priority = self._calculate_vnf_priority(req, dc.id)
                candidates.append((priority, req))
        
        if not candidates:
            # Không có request nào cần VNF này
            return config.REWARD_INVALID, False
        
        # Sort theo priority giảm dần
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_req = candidates[0][1]
        
        # 2. Tìm hoặc tạo VNF instance
        vnf_instance = dc.get_idle_vnf(vnf_name)
        
        if vnf_instance is None:
            # Cần cài đặt VNF mới
            if not dc.has_resources(vnf_name):
                return config.REWARD_INVALID, False
            
            # Tiêu thụ tài nguyên và tạo VNF mới
            dc.consume_resources(vnf_name)
            vnf_instance = VNFInstance(vnf_name, dc.id)
            dc.installed_vnfs.append(vnf_instance)
        
        # 3. Tính delay
        prop_delay = 0.0
        last_dc = best_req.get_last_placed_dc()
        if last_dc is not None and last_dc != dc.id:
            prop_delay = self.topology.get_propagation_delay(last_dc, dc.id)
        
        # Processing time từ spec (đơn vị: ms)
        proc_time = config.VNF_SPECS[vnf_name]['proc_time']
        waiting_time = 0.0
        
        # 4. Assign VNF với processing time
        vnf_instance.assign(best_req.id, proc_time, waiting_time)
        
        # 5. Advance chain (lưu lại delay info và VNF instance)
        total_proc_delay = proc_time + waiting_time
        best_req.advance_chain(dc.id, prop_delay, total_proc_delay, vnf_instance)
        
        # 6. Kiểm tra hoàn thành (check sau khi advance)
        best_req.check_completion()
        
        if best_req.is_completed:
            self.accepted_count += 1
            return config.REWARD_SATISFIED, True
        
        # Partial reward cho việc tiến bộ trong chain
        return config.REWARD_SATISFIED / 2.0, False

    def _handle_uninstall(self, dc, vnf_name):
        """Xử lý action Uninstall"""
        # Kiểm tra xem VNF này có được cần bởi request nào không
        is_needed = any(r.get_next_vnf() == vnf_name for r in self.manager.active_requests)
        
        if is_needed:
            # VNF đang cần cho request trong tương lai
            return config.REWARD_UNINSTALL_NEEDED, False
        
        # Kiểm tra xem có VNF idle để uninstall không
        vnf_to_remove = None
        for vnf in dc.installed_vnfs:
            if vnf.vnf_type == vnf_name and vnf.is_idle():
                vnf_to_remove = vnf
                break
        
        if vnf_to_remove:
            dc.installed_vnfs.remove(vnf_to_remove)
            dc.release_resources(vnf_name)
            return 0.0, False  # Neutral reward
        
        # Không thể uninstall (VNF đang busy hoặc không tồn tại)
        return config.REWARD_INVALID, False

    def _calculate_vnf_priority(self, req, dc_id):
        """
        Tính priority của VNF để chọn request phù hợp nhất
        
        P = P1 + P2 + P3
        - P1: elapsed_time - max_delay (càng gần deadline càng cao)
        - P2: Affinity (VNF trước đó có ở cùng DC không)
        - P3: Urgency (nếu remaining time < threshold)
        """
        # P1: Time priority
        p1 = req.elapsed_time - req.max_delay
        
        # P2: Affinity
        p2 = 0.0
        last_dc = req.get_last_placed_dc()
        if last_dc is not None:
            if last_dc == dc_id:
                p2 = config.PRIORITY_P2_SAME_DC
            else:
                p2 = config.PRIORITY_P2_DIFF_DC
        
        # P3: Urgency
        remaining_time = req.get_remaining_time()
        p3 = 0.0
        if remaining_time < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining_time + config.EPSILON_SMALL)
        
        return p1 + p2 + p3