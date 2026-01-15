import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=5):
        self.physical_graph = physical_graph
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        self.static_delays = dict(nx.all_pairs_dijkstra_path_length(self.physical_graph, weight='delay'))

    def allocate_bandwidth(self, src, dst, bw_demand):
        if src == dst:
            return True, [src], 0.0

        def weight_function(u, v, d):
            bw_capacity = d.get('capacity', 1000.0)
            bw_current = d.get('bw', 0.0)
            delay = d.get('delay', 1.0)
            
            if bw_current < bw_demand:
                return float('inf')
            
            usage_ratio = (bw_capacity - bw_current) / bw_capacity
            congestion_penalty = 1.0 + (usage_ratio * 3.0) ** 2 
            return delay * congestion_penalty

        try:
            path = nx.dijkstra_path(self.physical_graph, src, dst, weight=weight_function)
            
            total_real_delay = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge = self.physical_graph[u][v]
                edge['bw'] -= bw_demand
                total_real_delay += edge['delay']
                
            return True, path, total_real_delay

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, None, float('inf')

    def check_connectivity(self, src, dst, bw_demand):
        """Kiểm tra xem có đường đi với ĐỦ BĂNG THÔNG không."""
        if src == dst: return True

        def filter_edge(u, v):
            # Chỉ đi qua cạnh có bw còn lại >= demand
            return self.physical_graph[u][v].get('bw', 0) >= bw_demand
        
        try:
            view = nx.subgraph_view(self.physical_graph, filter_edge=filter_edge)
            return nx.has_path(view, src, dst)
        except:
            return False

    def release_bandwidth_on_path(self, path, bw_amount):
        if not path or len(path) < 2:
            return
        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                edge['bw'] = min(edge['capacity'], edge['bw'] + bw_amount)

    def get_estimated_delay(self, src, dst):
        if src == dst: return 0.0
        return self.static_delays.get(src, {}).get(dst, float('inf'))