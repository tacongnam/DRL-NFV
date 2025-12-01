import numpy as np
from config import *

def calculate_shortest_path(graph, source, destination, bandwidth_req, link_bw):
    n = len(graph)
    dist = [float('inf')] * n
    parent = [-1] * n
    visited = [False] * n
    dist[source] = 0
    
    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i
        
        if dist[u] == float('inf'):
            break
            
        visited[u] = True
        
        for v in range(n):
            if graph[u][v] > 0 and link_bw[u][v] >= bandwidth_req:
                if dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]
                    parent[v] = u
    
    if dist[destination] == float('inf'):
        return [], float('inf')
    
    path = []
    current = destination
    while current != -1:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    return path, dist[destination]

def calculate_propagation_delay(distance_km):
    return distance_km / (SPEED_OF_LIGHT / 1000)

def normalize_state(state, max_val=100):
    return np.clip(state / max_val, 0, 1)

def create_distance_matrix(num_dcs, min_dist=50, max_dist=500):
    matrix = np.zeros((num_dcs, num_dcs))
    for i in range(num_dcs):
        for j in range(i+1, num_dcs):
            dist = np.random.uniform(min_dist, max_dist)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix

def create_dc_resources(num_dcs):
    dcs = []
    for _ in range(num_dcs):
        cpu = np.random.uniform(*DC_CONFIG['cpu_range'])
        dcs.append({
            'cpu': cpu,
            'ram': DC_CONFIG['ram_capacity'],
            'storage': DC_CONFIG['storage_capacity'],
            'cpu_used': 0,
            'ram_used': 0,
            'storage_used': 0,
            'installed_vnfs': {vnf: 0 for vnf in VNF_TYPES},
            'allocated_vnfs': {vnf: 0 for vnf in VNF_TYPES}
        })
    return dcs

def generate_sfc_requests(request_count_range=(1, 3)):
    requests = []
    num_request_types = np.random.randint(*request_count_range)
    
    for _ in range(num_request_types):
        sfc_type = np.random.choice(SFC_TYPES)
        characteristics = SFC_CHARACTERISTICS[sfc_type]
        bundle_size = np.random.randint(*characteristics['bundle_size'])
        
        for _ in range(bundle_size):
            requests.append({
                'type': sfc_type,
                'chain': characteristics['chain'].copy(),
                'bandwidth': characteristics['bw'],
                'delay_tolerance': characteristics['delay'],
                'creation_time': 0,
                'allocated_vnfs': [],
                'status': 'pending'
            })
    
    return requests