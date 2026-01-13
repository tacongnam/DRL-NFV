import networkx as nx

class TopologyManager:
    def __init__(self, physical_graph, k_paths=5): # k_paths giữ lại để tương thích interface, nhưng không dùng tính tĩnh nữa
        self.physical_graph = physical_graph
        # Chỉ lưu danh sách server để tiện truy xuất
        self.servers = [n for n, d in self.physical_graph.nodes(data=True) if d.get('server', False)]
        
        # Cache khoảng cách tĩnh để dùng làm heuristic ước lượng (không dùng để routing thật)
        self.static_delays = dict(nx.all_pairs_dijkstra_path_length(self.physical_graph, weight='delay'))

    def allocate_bandwidth(self, src, dst, bw_demand):
        if src == dst:
            return True, [src], 0.0

        # Hàm tính trọng số động: Delay * (Hệ số tắc nghẽn)
        def weight_function(u, v, d):
            # Lấy thông tin cạnh
            bw_capacity = d.get('capacity', 1000.0) # Capacity gốc
            bw_current = d.get('bw', 0.0)           # Available BW hiện tại
            delay = d.get('delay', 1.0)
            
            # Nếu không đủ băng thông, trả về vô cực (coi như ngắt)
            if bw_current < bw_demand:
                return float('inf')
            
            # Tính tỷ lệ sử dụng: (Capacity - Available) / Capacity
            usage_ratio = (bw_capacity - bw_current) / bw_capacity
            
            # Penalty: Nếu đường càng đầy, "chi phí" đi qua càng đắt
            # Ví dụ: usage=0% -> cost=delay; usage=90% -> cost=delay * 10
            congestion_penalty = 1.0 + (usage_ratio * 3.0) ** 2 
            
            return delay * congestion_penalty

        try:
            # Dùng Dijkstra với hàm trọng số tùy chỉnh
            # NetworkX tự động gọi weight_function(u, v, edge_data)
            path = nx.dijkstra_path(self.physical_graph, src, dst, weight=weight_function)
            
            # Tính lại Delay thực tế (không có penalty) để trả về
            total_real_delay = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge = self.physical_graph[u][v]
                edge['bw'] -= bw_demand # Trừ băng thông
                total_real_delay += edge['delay']
                
            return True, path, total_real_delay

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