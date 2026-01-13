import random
import numpy as np
import networkx as nx
import config

class DataGenerator:
    @staticmethod
    def generate_scenario(num_nodes_range=(30, 50),  # Mạng trung bình-lớn để train nhanh hơn 100 node
                          server_ratio=0.2,           # 20% server
                          num_vnf_types=config.MAX_VNF_TYPES,
                          num_requests_range=(80, 150)): # Request dày đặc
        
        # 1. Barabasi-Albert Graph (Scale-Free)
        num_nodes = random.randint(*num_nodes_range)
        G = nx.barabasi_albert_graph(num_nodes, 3) # Attachment=3 để mạng dày hơn chút
        
        all_nodes = list(G.nodes())
        num_servers = int(num_nodes * server_ratio)
        server_ids = set(random.sample(all_nodes, num_servers))
        
        nodes_data = {}
        for n in all_nodes:
            if n in server_ids:
                nodes_data[str(n)] = {
                    "server": True,
                    # Tài nguyên giới hạn (Giống file Hard)
                    "c_v": random.uniform(30.0, 60.0),
                    "r_v": random.uniform(30.0, 60.0),
                    "h_v": random.uniform(50.0, 100.0),
                    "d_v": round(random.uniform(0.1, 0.3), 2),
                    "cost_c": 100, "cost_r": 50, "cost_h": 10
                }
            else:
                nodes_data[str(n)] = {"server": False}

        edges_data = []
        for u, v in G.edges():
            edges_data.append({
                "u": u, "v": v,
                "b_l": random.uniform(50.0, 150.0), # Băng thông hẹp
                "d_l": round(random.uniform(0.01, 0.05), 3)
            })
            
        vnf_specs = [{"c_f": 2.0, "r_f": 1.0, "h_f": 1.0, "d_f": {}} for _ in range(num_vnf_types)]
        
        # Requests dồn dập (0-50ms)
        requests = []
        arrival_times = np.sort(np.random.randint(0, 50, random.randint(*num_requests_range)))
        
        for t in arrival_times:
            src, dst = random.sample(all_nodes, 2)
            chain = [random.randint(0, num_vnf_types-1) for _ in range(random.randint(2, 4))]
            
            # Deadline cực gắt (Hard)
            proc_time = len(chain) * 3.0 # Giả sử xử lý 3ms/vnf
            prop_delay = 3.0             # Giả sử truyền dẫn 3ms
            d_max = round((proc_time + prop_delay) * random.uniform(1.1, 1.4), 1)
            
            requests.append({
                "T": int(t), "st_r": src, "d_r": dst, "F_r": chain, 
                "b_r": random.uniform(1.0, 5.0), "d_max": d_max
            })
            
        return {"V": nodes_data, "E": edges_data, "F": vnf_specs, "R": requests}