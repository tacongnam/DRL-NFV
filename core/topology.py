import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=5):
        self.physical_graph = physical_graph
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        # Cache static delays để không tính lại nhiều lần
        self.static_delays = dict(nx.all_pairs_dijkstra_path_length(self.physical_graph, weight='delay'))

    def allocate_bandwidth(self, src, dst, bw_demand):
        if src == dst:
            return True, [src], 0.0

        # FIX: filter_edge chỉ nhận (u, v), không nhận d
        def filter_edge(u, v):
            # Truy cập dữ liệu cạnh trực tiếp từ đồ thị gốc
            return self.physical_graph[u][v].get('bw', 0.0) >= bw_demand

        try:
            # Tạo view đồ thị chỉ gồm các cạnh đủ băng thông
            view = nx.subgraph_view(self.physical_graph, filter_edge=filter_edge)
            
            # Tìm đường ngắn nhất trên view này dựa trên DELAY
            path = nx.shortest_path(view, source=src, target=dst, weight='delay')
            
            # Tính toán và trừ băng thông
            total_real_delay = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge = self.physical_graph[u][v]
                edge['bw'] -= bw_demand
                total_real_delay += edge['delay']
                
            return True, path, total_real_delay

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, None, float('inf')

    def check_connectivity(self, src, dst, bw_demand):
        """Kiểm tra nhanh xem có đường đi hay không."""
        if src == dst: return True
        
        # Optimization: Kiểm tra nhanh nếu src bị cô lập (không có cạnh ra đủ BW)
        try:
            # Check nhanh các cạnh nối trực tiếp với src
            src_has_link = False
            for n in self.physical_graph[src]:
                if self.physical_graph[src][n].get('bw', 0) >= bw_demand:
                    src_has_link = True
                    break
            if not src_has_link: return False
        except KeyError:
            return False

        # Filter edges cho view
        def filter_edge(u, v):
            return self.physical_graph[u][v].get('bw', 0) >= bw_demand
            
        try:
            view = nx.subgraph_view(self.physical_graph, filter_edge=filter_edge)
            return nx.has_path(view, src, dst)
        except:
            return False

    def release_bandwidth_on_path(self, path, bw_amount):
        if not path or len(path) < 2: return
        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                # Trả lại băng thông nhưng không vượt quá capacity gốc
                edge['bw'] = min(edge['capacity'], edge['bw'] + bw_amount)

    def get_estimated_delay(self, src, dst):
        if src == dst: return 0.0
        return self.static_delays.get(src, {}).get(dst, float('inf'))