import networkx as nx

def constrained_shortest_path(G, src, dst, bw_demand):
    """
    Tối ưu hóa: Không copy đồ thị, lọc cạnh trực tiếp trong lúc chạy Dijkstra.
    """
    if src == dst:
        return [src], 0.0, float('inf')

    # Định nghĩa hàm trọng số tùy chỉnh
    # Nếu không đủ băng thông, coi như không có cạnh (trả về None hoặc inf)
    def weight_func(u, v, d):
        edge_bw = d.get('bw', 0)
        if edge_bw < bw_demand:
            return None  # NetworkX sẽ bỏ qua cạnh này
        return d.get('delay', 0.0)

    try:
        # Sử dụng nx.dijkstra_path để kiểm soát chặt chẽ weight function
        path = nx.dijkstra_path(G, src, dst, weight=weight_func)
        
        # Tính toán kết quả trong 1 lần duyệt path
        total_delay = 0.0
        min_bw = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            data = G[u][v]
            total_delay += data.get('delay', 0.0)
            if data['bw'] < min_bw:
                min_bw = data['bw']
        
        return path, total_delay, min_bw
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None, None

def allocate_bandwidth_on_path(G, path, bw_amount):
    """Giảm băng thông trực tiếp trên đồ thị gốc"""
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        G[u][v]['bw'] -= bw_amount

def release_bandwidth_on_path(G, path, bw_amount):
    """Hoàn trả băng thông, sử dụng 'capacity' để giới hạn"""
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = G[u][v]
        # Đảm bảo không vượt quá dung lượng vật lý ban đầu
        # Nếu không có 'capacity', lấy giá trị hiện tại + bw_amount
        limit = data.get('capacity', float('inf'))
        data['bw'] = min(data['bw'] + bw_amount, limit)