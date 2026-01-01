import networkx as nx
import numpy as np
from core.routing import constrained_shortest_path, allocate_bandwidth_on_path, release_bandwidth_on_path

class TopologyManager:
    def __init__(self, graph=None):
        self.graph = graph if graph is not None else nx.Graph()
        self.num_nodes = self.graph.number_of_nodes()
        
        # Cache shortest paths between DataCenters
        self._path_cache = {}
        self._delay_cache = {}
        self._min_bw_cache = {}
        self._precompute_paths()
    
    def _precompute_paths(self):
        """Precompute shortest paths between all DataCenter pairs"""
        servers = [n for n, d in self.graph.nodes(data=True) if d.get('server', False)]
        
        for src in servers:
            for dst in servers:
                if src != dst:
                    path, delay, min_bw = constrained_shortest_path(self.graph, src, dst, 0.0)
                    if path:
                        self._path_cache[(src, dst)] = path
                        self._delay_cache[(src, dst)] = delay
                        self._min_bw_cache[(src, dst)] = min_bw
    
    def get_propagation_delay(self, dc_i, dc_j, bw_demand=0.0):
        if dc_i == dc_j:
            return 0.0
        
        key = (dc_i, dc_j)
        if key in self._delay_cache:
            cached_min_bw = self._min_bw_cache[key]
            if cached_min_bw >= bw_demand:
                return self._delay_cache[key]
        
        # Fallback: recompute
        path, delay, _ = constrained_shortest_path(self.graph, dc_i, dc_j, bw_demand)
        return delay if delay is not None else float('inf')
    
    def consume_bandwidth(self, dc_i, dc_j, bw_required):
        if dc_i == dc_j:
            return True, [dc_i], 0.0
        
        path, delay, min_bw = constrained_shortest_path(self.graph, dc_i, dc_j, bw_required)
        
        if path is None:
            return False, None, None
        
        allocate_bandwidth_on_path(self.graph, path, bw_required)
        
        # Invalidate cache for affected pairs
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            for key in list(self._min_bw_cache.keys()):
                if (u in self._path_cache.get(key, [])) or (v in self._path_cache.get(key, [])):
                    self._min_bw_cache[key] = max(0, self._min_bw_cache[key] - bw_required)
        
        return True, path, delay
    
    def release_bandwidth(self, dc_i, dc_j, bw_amount):
        if dc_i == dc_j:
            return
        
        key = (dc_i, dc_j)
        path = self._path_cache.get(key)
        
        if path:
            release_bandwidth_on_path(self.graph, path, bw_amount)
            
            # Update cache
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                for k in list(self._min_bw_cache.keys()):
                    if (u in self._path_cache.get(k, [])) or (v in self._path_cache.get(k, [])):
                        capacity = self.graph[u][v].get('capacity', float('inf'))
                        self._min_bw_cache[k] = min(capacity, self._min_bw_cache[k] + bw_amount)
    
    def get_shortest_path_dcs(self, source, destination, bw_demand=0.0):
        if source == destination:
            return [source]
        
        key = (source, destination)
        if key in self._path_cache and bw_demand == 0.0:
            return self._path_cache[key]
        
        path, _, _ = constrained_shortest_path(self.graph, source, destination, bw_demand)
        return path if path else [source, destination]