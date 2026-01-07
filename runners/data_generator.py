import random

class DataGenerator:
    @staticmethod
    def generate_scenario(num_dcs_range=(2, 10), num_switches_range=(2, 6), num_vnf_types=2, num_requests_range=(5, 20)):
        num_dcs = random.randint(*num_dcs_range)
        num_switches = random.randint(*num_switches_range)
        total_nodes = num_dcs + num_switches
        
        nodes = {}
        for i in range(num_dcs):
            nodes[str(i)] = {
                "server": True,
                "c_v": random.randint(20, 100),
                "r_v": random.randint(30, 200),
                "h_v": random.randint(20, 150),
                "d_v": round(random.uniform(0.01, 0.5), 2),
                "cost_c": random.randint(80, 150),
                "cost_r": random.randint(50, 100),
                "cost_h": random.randint(15, 30)
            }
        
        for i in range(num_dcs, total_nodes):
            nodes[str(i)] = {"server": False}
        
        edges = []
        all_nodes = list(range(total_nodes))
        
        for i in range(num_dcs):
            targets = random.sample([n for n in all_nodes if n != i], min(3, total_nodes - 1))
            for j in targets:
                if i < j:
                    edges.append({
                        "u": i, "v": j,
                        "b_l": random.randint(50, 200),
                        "d_l": round(random.uniform(0.01, 0.1), 2)
                    })
        
        for i in range(num_dcs, total_nodes):
            targets = random.sample([n for n in all_nodes if n != i], min(2, total_nodes - 1))
            for j in targets:
                if i < j:
                    edges.append({
                        "u": i, "v": j,
                        "b_l": random.randint(50, 150),
                        "d_l": round(random.uniform(0.01, 0.08), 2)
                    })
        
        vnf_specs = []
        for _ in range(num_vnf_types):
            vnf_specs.append({
                "c_f": round(random.uniform(0.8, 1.5), 1),
                "r_f": round(random.uniform(0.8, 1.5), 1),
                "h_f": round(random.uniform(0.7, 1.2), 1),
                "d_f": {str(i): round(random.uniform(0.03, 0.08), 2) for i in range(random.randint(2, 4))}
            })
        
        num_requests = random.randint(*num_requests_range)
        requests = []
        for i in range(num_requests):
            chain_length = random.randint(1, min(3, num_vnf_types))
            requests.append({
                "T": random.randint(1, 50),
                "st_r": random.randint(0, total_nodes - 1),
                "d_r": random.randint(0, total_nodes - 1),
                "F_r": [random.randint(0, num_vnf_types - 1) for _ in range(chain_length)],
                "b_r": round(random.uniform(0.5, 3.0), 1),
                "d_max": round(random.uniform(20.0, 150.0), 1)
            })
        
        return {"V": nodes, "E": edges, "F": vnf_specs, "R": requests}