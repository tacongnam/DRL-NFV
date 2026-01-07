import json
import networkx as nx
import config
from core import DataCenter

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    graph = nx.Graph()
    dcs = []
    
    for node_id_str, node_data in data["V"].items():
        node_id = int(node_id_str)
        is_server = node_data.get("server", False)
        
        graph.add_node(
            node_id,
            server=is_server,
            cpu=node_data.get("c_v", 0),
            ram=node_data.get("r_v", 0),
            delay=node_data.get("d_v", 0)
        )
        
        if is_server:
            dc = DataCenter(
                dc_id=node_id,
                cpu=node_data.get("c_v"),
                ram=node_data.get("r_v"),
                storage=node_data.get("h_v"),
                delay=node_data.get("d_v", 0.0),
                cost_c=node_data.get("cost_c", 1.0),
                cost_h=node_data.get("cost_h", 1.0),
                cost_r=node_data.get("cost_r", 1.0)
            )
            dcs.append(dc)
    
    for link in data["E"]:
        u, v = int(link["u"]), int(link["v"])
        bw = link.get("b_l", config.LINK_BW_CAPACITY)
        delay = link.get("d_l", 1.0)
        
        graph.add_edge(u, v, bw=bw, capacity=bw, delay=delay)
    
    vnf_specs = {}
    for idx, vnf in enumerate(data["F"]):
        vnf_specs[idx] = {
            "cpu": vnf.get("c_f", 1.0),
            "ram": vnf.get("r_f", 1.0),
            "storage": vnf.get("h_f", 1.0),
            "startup_time": {int(k): v for k, v in vnf.get("d_f", {}).items()}
        }
    
    requests = []
    for idx, req in enumerate(data["R"]):
        requests.append({
            "id": idx,
            "arrival_time": req.get("T", 0),
            "source": req["st_r"],
            "destination": req["d_r"],
            "vnf_chain": req["F_r"],
            "bandwidth": req.get("b_r", 1.0),
            "max_delay": req.get("d_max", 100.0),
            "type": req.get("type", "Unknown")
        })
    
    dcs.sort(key=lambda x: x.id)
    
    return graph, dcs, requests, vnf_specs