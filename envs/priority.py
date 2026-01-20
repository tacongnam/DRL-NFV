class PriorityManager:
    def __init__(self, topology_manager):
        self.topology = topology_manager
    
    def get_dc_priority_order(self, dcs, active_requests):
        server_dcs = [dc for dc in dcs if dc.is_server]
        
        if not active_requests:
            return [dc.id for dc in server_dcs]
        
        min_delay_req = min(active_requests, key=lambda r: r.max_delay)
        source_dc = min_delay_req.source
        dest_dc = min_delay_req.destination
        
        path_dcs = self.topology.get_shortest_path_dcs(source_dc, dest_dc)
        
        priority_scores = {}
        
        for dc in server_dcs:
            dc_id = dc.id
            if dc_id == source_dc:
                priority_scores[dc_id] = 1000.0
            elif dc_id in path_dcs:
                path_index = path_dcs.index(dc_id)
                priority_scores[dc_id] = 500.0 - path_index * 10
            else:
                priority_scores[dc_id] = 0.0
        
        sorted_dcs = sorted(priority_scores.keys(), 
                           key=lambda x: priority_scores[x], 
                           reverse=True)
        
        return sorted_dcs