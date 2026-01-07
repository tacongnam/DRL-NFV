import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=3):
        self.physical_graph = physical_graph
        self.k = k_paths
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        self._precompute_k_shortest_paths()
        
    def _precompute_k_shortest_paths(self):
        self.path_table = {}
        for src in self.servers:
            for dst in self.servers:
                if src != dst:
                    try:
                        paths = list(nx.shortest_simple_paths(
                            self.physical_graph, src, dst, weight='delay'
                        ))[:self.k]
                        self.path_table[(src, dst)] = [
                            {
                                'path': p,
                                'delay': sum(self.physical_graph[u][v]['delay'] 
                                           for u, v in zip(p[:-1], p[1:]))
                            }
                            for p in paths
                        ]
                    except nx.NetworkXNoPath:
                        self.path_table[(src, dst)] = []
    
    def allocate_bandwidth(self, src, dst, bw_demand):
        if src == dst:
            return True, [src], 0.0
        
        candidates = self.path_table.get((src, dst), [])
        
        for cand in candidates:
            path = cand['path']
            min_bw = min(self.physical_graph[u][v]['bw'] 
                        for u, v in zip(path[:-1], path[1:]))
            if min_bw >= bw_demand:
                for u, v in zip(path[:-1], path[1:]):
                    self.physical_graph[u][v]['bw'] -= bw_demand
                return True, path, cand['delay']
        
        return False, None, float('inf')

    def get_estimated_delay(self, src, dst):
        if src == dst:
            return 0.0
        try:
            return nx.shortest_path_length(self.physical_graph, src, dst, weight='delay')
        except nx.NetworkXNoPath:
            return float('inf')
    
    def release_bandwidth_on_path(self, path, bw_amount):
        if not path or len(path) < 2:
            return

        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                limit = edge.get('capacity', float('inf'))
                edge['bw'] = min(limit, edge['bw'] + bw_amount)