# DRL/core/topology.py
import networkx as nx
from core.routing import constrained_shortest_path, allocate_bandwidth_on_path, release_bandwidth_on_path

class TopologyManager:
    """Manage physical network graph and routing"""
    
    def __init__(self, graph=None):
        """
        Initialize with NetworkX graph from data loader.
        
        Args:
            graph: NetworkX graph with nodes (cpu, ram, storage, delay) 
                   and edges (bw, delay, capacity)
        """
        self.graph = graph if graph is not None else nx.Graph()
        self.num_dcs = self.graph.number_of_nodes()
    
    def get_propagation_delay(self, dc_i, dc_j, bw_demand=0.0):
        """
        Get propagation delay via shortest constrained path.
        
        Args:
            dc_i: Source DC ID
            dc_j: Destination DC ID
            bw_demand: Required bandwidth (default 0 for query only)
        
        Returns:
            Delay in ms, or None if no valid path
        """
        path, delay, _ = constrained_shortest_path(self.graph, dc_i, dc_j, bw_demand)
        return delay
    
    def get_shortest_path_dcs(self, source, destination, bw_demand=0.0):
        """
        Get shortest delay path with bandwidth constraint.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            bw_demand: Required bandwidth
        
        Returns:
            List of node IDs on path, or [source, destination] if no path
        """
        path, _, _ = constrained_shortest_path(self.graph, source, destination, bw_demand)
        
        if path is None:
            return [source, destination]
        
        return path
    
    def consume_bandwidth(self, dc_i, dc_j, bw_required):
        """
        Consume bandwidth on shortest path between two DCs.
        
        Args:
            dc_i: Source DC ID
            dc_j: Destination DC ID
            bw_required: Bandwidth to consume
        
        Returns:
            (success, path, delay) tuple
        """
        path, delay, min_bw = constrained_shortest_path(self.graph, dc_i, dc_j, bw_required)
        
        if path is None:
            return False, None, None
        
        allocate_bandwidth_on_path(self.graph, path, bw_required)
        return True, path, delay
    
    def release_bandwidth(self, dc_i, dc_j, bw_amount):
        """
        Release bandwidth on shortest path (best effort).
        Note: May not match exact allocation path if topology changed.
        """
        path, _, _ = constrained_shortest_path(self.graph, dc_i, dc_j, 0.0)
        
        if path is not None:
            release_bandwidth_on_path(self.graph, path, bw_amount)
    
    def get_path_metrics(self, path):
        """
        Calculate metrics for a given path.
        
        Args:
            path: List of node IDs
        
        Returns:
            (total_delay, min_bw, hop_count) tuple
        """
        if path is None or len(path) <= 1:
            return 0.0, float('inf'), 0
        
        total_delay = 0.0
        min_bw = float('inf')
        hop_count = len(path) - 1
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            total_delay += edge_data.get('delay', 0.0)
            min_bw = min(min_bw, edge_data.get('bw', 0))
        
        return total_delay, min_bw, hop_count