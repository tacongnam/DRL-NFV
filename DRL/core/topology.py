# spaces/topology.py
import numpy as np
from DRL import config

class TopologyManager:
    """Quản lý topology mạng và tính toán propagation delay"""
    
    def __init__(self, num_dcs, delay_matrix=None, bw_matrix=None):
        self.num_dcs = num_dcs
        # Ma trận delay
        if delay_matrix is not None:
            self.delay_matrix = delay_matrix
        else:
            self.delay_matrix = self._generate_delay_matrix()
        # Ma trận băng thông khả dụng
        if bw_matrix is not None:
            self.bw_matrix = bw_matrix
        else:
            self.bw_matrix = np.full((num_dcs, num_dcs), config.LINK_BW_CAPACITY, dtype=float)
            np.fill_diagonal(self.bw_matrix, 0)
    
    def _generate_delay_matrix(self):
        """
        Tạo ma trận delay ngẫu nhiên giữa các DC (dùng khi không có dữ liệu)
        Đơn vị: ms
        """
        W = np.zeros((self.num_dcs, self.num_dcs))
        
        for i in range(self.num_dcs):
            for j in range(i + 1, self.num_dcs):
                # Delay ngẫu nhiên
                delay = np.random.uniform(1, 100)
                W[i, j] = delay
                W[j, i] = delay
        
        return W
    
    def get_propagation_delay(self, dc_i, dc_j):
        """
        Lấy propagation delay giữa 2 DC từ delay_matrix
        
        Args:
            dc_i, dc_j: ID của 2 DC
        Returns:
            Propagation delay (ms)
        """
        if dc_i == dc_j:
            return 0.0
        
        return self.delay_matrix[dc_i, dc_j]
    
    def get_shortest_path_dcs(self, source, destination):
        """
        Tìm đường đi ngắn nhất từ source đến destination
        Trả về list các DC trên đường đi (bao gồm source và destination)
        
        Sử dụng thuật toán Dijkstra đơn giản
        """
        if source == destination:
            return [source]
        
        # Dijkstra's algorithm
        dist = [float('inf')] * self.num_dcs
        prev = [None] * self.num_dcs
        visited = [False] * self.num_dcs
        
        dist[source] = 0
        
        for _ in range(self.num_dcs):
            # Tìm node chưa visit có dist nhỏ nhất
            min_dist = float('inf')
            u = -1
            for i in range(self.num_dcs):
                if not visited[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i
            
            if u == -1:
                break
            
            visited[u] = True
            
            # Update neighbors
            for v in range(self.num_dcs):
                if not visited[v] and self.delay_matrix[u, v] > 0:
                    alt = dist[u] + self.delay_matrix[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
        
        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.insert(0, current)
            current = prev[current]
        
        if path[0] != source:
            return [source, destination]  # Fallback: direct path
        
        return path
    
    def consume_bandwidth(self, dc_i, dc_j, bw_required):
        """
        Tiêu thụ băng thông trên link
        Trả về True nếu thành công
        """
        if self.bw_matrix[dc_i, dc_j] >= bw_required:
            self.bw_matrix[dc_i, dc_j] -= bw_required
            self.bw_matrix[dc_j, dc_i] -= bw_required
            return True
        return False
    
    def release_bandwidth(self, dc_i, dc_j, bw_amount):
        """Giải phóng băng thông"""
        self.bw_matrix[dc_i, dc_j] += bw_amount
        self.bw_matrix[dc_j, dc_i] += bw_amount