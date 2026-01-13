import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=5): # k_paths giữ lại để tương thích interface, nhưng không dùng tính tĩnh nữa
        self.physical_graph = physical_graph
        # Chỉ lưu danh sách server để tiện truy xuất
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        
        # Cache khoảng cách tĩnh để dùng làm heuristic ước lượng (không dùng để routing thật)
        self.static_delays = dict(nx.all_pairs_dijkstra_path_length(self.physical_graph, weight='delay'))

    def allocate_bandwidth(self, src, dst, bw_demand):
        """
        Tìm đường đi động (Dynamic Routing) trên đồ thị đã lọc băng thông.
        Sử dụng Bidirectional Dijkstra để tối ưu tốc độ.
        """
        if src == dst:
            return True, [src], 0.0

        # 1. Định nghĩa bộ lọc: Chỉ đi qua các cạnh có bw >= demand
        # Hàm này cực nhẹ, chỉ so sánh số
        def filter_edge(u, v):
            return self.physical_graph[u][v]['bw'] >= bw_demand

        try:
            # 2. Tạo view ảo (Zero-copy), O(1) time
            view = nx.subgraph_view(self.physical_graph, filter_edge=filter_edge)
            
            # 3. Dùng Bidirectional Dijkstra: Nhanh gấp đôi Dijkstra thường
            # Tìm đường đi ngắn nhất dựa trên Delay
            _, path = nx.bidirectional_dijkstra(view, src, dst, weight='delay')
            
            # 4. Tính toán và trừ băng thông
            total_delay = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge = self.physical_graph[u][v] # Truy cập edge gốc
                edge['bw'] -= bw_demand
                total_delay += edge['delay']
                
            return True, path, total_delay

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, None, float('inf')

    def check_connectivity(self, src, dst, bw_demand):
        """
        Kiểm tra nhanh kết nối (Dùng cho Action Masking).
        Sử dụng BFS (has_path) thay vì Dijkstra.
        """
        if src == dst: return True

        def filter_edge(u, v):
            return self.physical_graph[u][v]['bw'] >= bw_demand
        
        try:
            view = nx.subgraph_view(self.physical_graph, filter_edge=filter_edge)
            # has_path dùng BFS/DFS, nhanh hơn nhiều so với tìm đường ngắn nhất
            return nx.has_path(view, src, dst)
        except:
            return False

    def release_bandwidth_on_path(self, path, bw_amount):
        """Trả lại băng thông."""
        if not path or len(path) < 2:
            return
            
        for u, v in zip(path[:-1], path[1:]):
            if self.physical_graph.has_edge(u, v):
                edge = self.physical_graph[u][v]
                # Cập nhật trực tiếp vào đồ thị gốc
                edge['bw'] = min(edge['capacity'], edge['bw'] + bw_amount)

    def get_estimated_delay(self, src, dst):
        """Trả về delay tĩnh đã tính trước (O(1)). Dùng cho Observer tính toán state."""
        if src == dst: return 0.0
        return self.static_delays.get(src, {}).get(dst, float('inf'))