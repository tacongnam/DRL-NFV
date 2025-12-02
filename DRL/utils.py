import numpy as np
import networkx as nx
from config import LIGHT_SPEED, PRIORITY_CONFIG

def create_network_topology(num_dcs, dc_positions=None):
    G = nx.Graph()
    if dc_positions is None:
        dc_positions = np.random.rand(num_dcs, 2) * 1000
    
    for i in range(num_dcs):
        G.add_node(i, pos=dc_positions[i])
    
    for i in range(num_dcs):
        for j in range(i + 1, num_dcs):
            dist = np.linalg.norm(dc_positions[i] - dc_positions[j])
            G.add_edge(i, j, weight=dist, available_bw=1000)
    
    return G

def calculate_propagation_delay(G, path):
    delay = 0
    for i in range(len(path) - 1):
        dist = G[path[i]][path[i+1]]['weight']
        delay += (dist * 1000) / LIGHT_SPEED * 1000
    return delay

def get_shortest_path_with_bw(G, source, dest, required_bw):
    try:
        all_paths = list(nx.all_simple_paths(G, source, dest))
        valid_paths = []
        for path in all_paths:
            valid = True
            for i in range(len(path) - 1):
                if G[path[i]][path[i+1]]['available_bw'] < required_bw:
                    valid = False
                    break
            if valid:
                valid_paths.append(path)
        
        if not valid_paths:
            return None
        
        return min(valid_paths, key=lambda p: sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)))
    except:
        return None

def calculate_priority_p1(elapsed_time, e2e_delay):
    return elapsed_time - e2e_delay

def calculate_priority_p2(vnf, dc_id, sfc_allocated_dcs):
    sfc_id = vnf['sfc_id']
    allocated_dcs = sfc_allocated_dcs.get(sfc_id, [])
    
    if dc_id in allocated_dcs:
        return 10
    elif allocated_dcs:
        return -5
    return 0

def calculate_priority_p3(elapsed_time, e2e_delay):
    remaining = e2e_delay - elapsed_time
    if remaining < PRIORITY_CONFIG['urgency_threshold']:
        return PRIORITY_CONFIG['urgency_constant'] / (remaining + PRIORITY_CONFIG['epsilon'])
    return 0

def check_resource_availability(dc_resources, vnf_type, vnf_specs):
    required = vnf_specs[vnf_type]
    return (dc_resources['cpu'] >= required['cpu'] and 
            dc_resources['ram'] >= required['ram'] and
            dc_resources['storage'] >= required['storage'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state1, state2, state3, action, reward, next_state1, next_state2, next_state3, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state1, state2, state3, action, reward, 
                                       next_state1, next_state2, next_state3, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        s1, s2, s3, a, r, ns1, ns2, ns3, d = zip(*batch)
        return (np.array(s1), np.array(s2), np.array(s3), np.array(a), 
                np.array(r), np.array(ns1), np.array(ns2), np.array(ns3), np.array(d))
    
    def __len__(self):
        return len(self.buffer)