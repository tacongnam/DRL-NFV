import networkx as nx
from itertools import islice

class TopologyManager:
    '''
    Quản lý đồ thị vật lý (DC + Switch) và cung cấp góc nhìn logic (DC-to-DC) cho Agent.
    '''

    def __init__(self, physical_graph, k_paths=3):
        self.physical_graph = physical_graph
        self.k = k_paths
        
        # Cache đường đi: Key=(src_id, dst_id), Value=List[{path, base_delay}]
        self.logical_routes = {}
        self._precompute_k_paths()
    
    def _precompute_k_paths(self):
        '''
        Tính trước K đường đi ngắn nhất (về delay) giữa các cặp Server (DC).
        '''
        servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]

        for src in servers:
            for dst in servers:
                if src == dst: 
                    self.logical_routes[(src, dst)] = [{'path': [src], 'base_delay': 0.0}]
                    continue
                
                try:
                    k_paths_gen = nx.shortest_simple_paths(self.physical_graph, src, dst, weight='delay')
                    k_paths = list(islice(k_paths_gen, self.k))

                    self.logical_routes[(src, dst)] = []
                    for path in k_paths:
                        link_delay = sum(self.physical_graph[u][v].get('delay', 0) for u, v in zip(path[:-1], path[1:]))
                        self.logical_routes[(src, dst)].append({
                            'path': path,
                            'base_delay': link_delay
                        })
                except nx.NetworkXNoPath:
                    self.logical_routes[(src, dst)] = []

    def get_estimated_delay(self, src, dst):
        """Trả về delay tốt nhất hiện có giữa 2 DC."""
        if src == dst: return 0.0
        candidates = self.logical_routes.get((src, dst), [])
        if not candidates: return float('inf')
        return candidates[0]['base_delay']

    def allocate_bandwidth(self, src, dst, bw_demand):
        '''
        Tìm đường đi khả thi và trừ BW.
        Returns: (success, path, delay)
        '''
        if src == dst:
            return True, [src], 0.0

        candidates = self.logical_routes.get((src, dst), [])

        for cand in candidates:
            path = cand['path']
            
            is_feasible = True
            for u, v in zip(path[:-1], path[1:]):
                current_bw = self.physical_graph[u][v].get('bw', 0)
                if current_bw < bw_demand:
                    is_feasible = False
                    break

            if is_feasible:
                for u, v in zip(path[:-1], path[1:]):
                    self.physical_graph[u][v]['bw'] -= bw_demand
                return True, path, cand['base_delay']
        
        return False, None, float('inf')
    
    def release_bandwidth_on_path(self, path, bw_amount):
        '''Hoàn trả băng thông.'''
        if not path or len(path) < 2: return

        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                limit = edge.get('capacity', float('inf'))
                edge['bw'] = min(limit, edge['bw'] + bw_amount)