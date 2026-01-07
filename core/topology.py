import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=5):
        self.physical_graph = physical_graph
        self.k = k_paths
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        self.all_nodes = list(self.physical_graph.nodes())
        self._precompute_server_paths()
        
    def _precompute_server_paths(self):
        self.path_table = {}
        for src in self.servers:
            for dst in self.servers:
                if src == dst:
                    self.path_table[(src, dst)] = [{'path': [src], 'delay': 0.0}]
                    continue
                    
                try:
                    paths = []
                    for path in nx.shortest_simple_paths(self.physical_graph, src, dst, weight='delay'):
                        paths.append(path)
                        if len(paths) >= self.k:
                            break
                    
                    self.path_table[(src, dst)] = [
                        {
                            'path': p,
                            'delay': sum(self.physical_graph[u][v]['delay'] 
                                       for u, v in zip(p[:-1], p[1:]))
                        }
                        for p in paths
                    ]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self.path_table[(src, dst)] = []

    def _find_best_path_runtime(self, src, dst, bw_demand):
        if src == dst:
            return [src], 0.0
        
        try:
            paths = []
            for path in nx.shortest_simple_paths(self.physical_graph, src, dst, weight='delay'):
                min_bw = min(self.physical_graph[u][v]['bw'] 
                           for u, v in zip(path[:-1], path[1:]))
                if min_bw >= bw_demand:
                    delay = sum(self.physical_graph[u][v]['delay'] 
                              for u, v in zip(path[:-1], path[1:]))
                    paths.append((path, delay))
                
                if len(paths) >= self.k:
                    break
            
            if paths:
                paths.sort(key=lambda x: x[1])
                return paths[0]
            
            return None, float('inf')
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')

    def get_estimated_delay(self, src, dst):
        if src == dst:
            return 0.0
        
        src_is_server = src in self.servers
        dst_is_server = dst in self.servers
        
        if src_is_server and dst_is_server:
            candidates = self.path_table.get((src, dst), [])
            if candidates:
                return candidates[0]['delay']
        
        try:
            return nx.shortest_path_length(self.physical_graph, src, dst, weight='delay')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def allocate_bandwidth(self, src, dst, bw_demand):
        if src == dst:
            return True, [src], 0.0
        
        src_is_server = src in self.servers
        dst_is_server = dst in self.servers
        
        if src_is_server and dst_is_server:
            candidates = self.path_table.get((src, dst), [])
        else:
            path, delay = self._find_best_path_runtime(src, dst, bw_demand)
            if path is None:
                return False, None, float('inf')
            candidates = [{'path': path, 'delay': delay}]
        
        if not candidates:
            return False, None, float('inf')
        
        for cand in candidates:
            path = cand['path']
            
            is_feasible = True
            
            for u, v in zip(path[:-1], path[1:]):
                if not self.physical_graph.has_edge(u, v):
                    is_feasible = False
                    break
                
                current_bw = self.physical_graph[u][v].get('bw', 0)
                
                if current_bw < bw_demand:
                    is_feasible = False
                    break
            
            if is_feasible:
                for u, v in zip(path[:-1], path[1:]):
                    self.physical_graph[u][v]['bw'] -= bw_demand
                return True, path, cand['delay']
        
        return False, None, float('inf')
    
    def release_bandwidth_on_path(self, path, bw_amount):
        if not path or len(path) < 2:
            return

        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                limit = edge.get('capacity', float('inf'))
                edge['bw'] = min(limit, edge['bw'] + bw_amount)