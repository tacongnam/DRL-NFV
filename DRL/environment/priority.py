# environment/priority.py
import numpy as np
import config

class PriorityManager:
    """Quản lý priority của DC cho việc iteration"""
    
    def __init__(self, topology_manager):
        self.topology = topology_manager
    
    def get_dc_priority_order(self, dcs, active_requests):
        """
        Tính priority order của các DC
        
        Priority logic:
        1. Highest: Source DC của request có E2E delay nhỏ nhất
        2. Medium: DC trên shortest path từ source đến destination
        3. Lowest: DC không nằm trên path
        
        Returns:
            List[int]: Danh sách DC IDs theo thứ tự priority giảm dần
        """
        if not active_requests:
            # Không có request, trả về thứ tự mặc định
            return list(range(len(dcs)))
        
        # 1. Tìm request có E2E delay constraint nhỏ nhất
        min_delay_req = min(active_requests, key=lambda r: r.max_delay)
        source_dc = min_delay_req.source
        dest_dc = min_delay_req.destination
        
        # 2. Tìm shortest path
        path_dcs = self.topology.get_shortest_path_dcs(source_dc, dest_dc)
        
        # 3. Tạo priority list
        priority_scores = {}
        
        for dc_id in range(len(dcs)):
            if dc_id == source_dc:
                # Highest priority: source DC
                priority_scores[dc_id] = 1000.0
            elif dc_id in path_dcs:
                # Medium priority: DC trên path
                # Càng gần source càng cao priority
                path_index = path_dcs.index(dc_id)
                priority_scores[dc_id] = 500.0 - path_index * 10
            else:
                # Low priority: DC không trên path
                priority_scores[dc_id] = 0.0
        
        # Sort theo priority giảm dần
        sorted_dcs = sorted(priority_scores.keys(), 
                           key=lambda x: priority_scores[x], 
                           reverse=True)
        
        return sorted_dcs