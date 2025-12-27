import json
import sys
import numpy as np
sys.path.append('..')

from DRL.core.dc import DataCenter
from DRL.core.topology import TopologyManager
from DRL import config

class Read_data:
    def __init__(self, PATH):
        with open(PATH) as f:
            self.data = json.load(f)
        
    def get_V(self):
        dc_list = []
        for key, value in self.data["V"].items():
            dc_id = int(key)
            if value["server"] == True:
                storage = value["h_v"]  # memory_cap maps to storage
                cpu = value["c_v"]
                ram = value["r_v"]
                delay = value["d_v"]
                cost_c = value["cost_c"]
                cost_h = value["cost_h"]
                cost_r = value["cost_r"]
                dc = DataCenter(dc_id, cpu=cpu, ram=ram, storage=storage, delay=delay, 
                               cost_c=cost_c, cost_h=cost_h, cost_r=cost_r, is_server=True)
                dc_list.append(dc)
            else:
                # Non-server node: cannot deploy VNFs, all attributes None
                dc = DataCenter(dc_id, is_server=False)
                dc_list.append(dc)
        return dc_list
    
    def get_E(self, num_dcs):
        # Initialize matrices with infinity delay and 0 bandwidth for missing edges
        delay_matrix = np.full((num_dcs, num_dcs), np.inf)
        bw_matrix = np.zeros((num_dcs, num_dcs))
        
        # Set diagonal to 0 delay and infinite bandwidth (same node)
        np.fill_diagonal(delay_matrix, 0)
        np.fill_diagonal(bw_matrix, np.inf)
        
        # Populate matrices from link data
        for link in self.data["E"]:
            u = int(link["u"])
            v = int(link["v"])
            bw_cap = link["b_l"]
            delay = link["d_l"]
            
            # Set delay (symmetric)
            delay_matrix[u, v] = delay
            delay_matrix[v, u] = delay
            
            # Set bandwidth capacity (symmetric)
            bw_matrix[u, v] = bw_cap
            bw_matrix[v, u] = bw_cap
        
        # Create TopologyManager with populated matrices
        topology = TopologyManager(num_dcs, delay_matrix=delay_matrix, bw_matrix=bw_matrix)
        return topology
    
    def get_F(self):
        """
        Parse VNF specifications from JSON data and update config.VNF_SPECS.
        
        Returns:
            dict: VNF specs indexed by position (int key), each containing:
                - 'cpu': CPU used (c_f)
                - 'ram': RAM used (r_f)
                - 'storage': storage used (h_f)
                - 'startup_time': dict of startup times per DC node (d_f)
        
        Note: Processing time for a VNF in a DC = startup_time[dc_id] + DC.delay
        """
        vnf_specs = {}
        for i in range(len(self.data["F"])):
            c_f = self.data["F"][i]["c_f"]
            r_f = self.data["F"][i]["r_f"]
            h_f = self.data["F"][i]["h_f"]
            
            # Parse startup times per DC node
            startup_time = {}
            for key, value in self.data["F"][i]["d_f"].items():
                startup_time[int(key)] = value
            
            vnf_specs[i] = {
                'cpu': c_f,
                'ram': r_f,
                'storage': h_f,
                'startup_time': startup_time  # startup time on each server node if deployed in that DC
            }
        
        # Update global VNF_SPECS in config
        config.update_vnf_specs(vnf_specs)
        return vnf_specs
    
    def get_R(self):
        """
        Parse SFC requests from JSON data.
        
        Returns:
            list: List of request dictionaries, each containing:
                - 'id': Request ID (sequential)
                - 'arrival_time': When request arrives (T)
                - 'source': Start node (st_r)
                - 'destination': Destination node (d_r)
                - 'vnf_chain': Ordered list of VNF type indices (F_r)
                - 'bandwidth': Bandwidth required (b_r)
                - 'max_delay': Maximum delay allowed (d_max)
        """
        r_list = []
        for req_id, r in enumerate(self.data["R"]):
            request_data = {
                'id': req_id,
                'arrival_time': r["T"],
                'source': r["st_r"],
                'destination': r["d_r"],
                'vnf_chain': r["F_r"],  # List of VNF type indices
                'bandwidth': r["b_r"],
                'max_delay': r["d_max"]
            }
            r_list.append(request_data)
        return r_list
    
    def get_info_network(self):
        """
        Get network information statistics.
        
        Returns:
            dict: Network statistics containing:
                - num_dcs: Total number of data centers
                - num_servers: Number of server nodes
                - num_vnf_types: Number of VNF types
                - num_requests: Number of requests
                - server_stats: Server resource statistics
                - vnf_stats: VNF resource statistics
                - topology_stats: Topology statistics
        """
        dc_list = self.get_V()
        topology = self.get_E(len(dc_list))
        vnf_specs = self.get_F()
        requests = self.get_R()
        
        # Server statistics
        servers = [dc for dc in dc_list if dc.is_server]
        if servers:
            server_stats = {
                'max_cpu': max(dc.cpu for dc in servers),
                'max_ram': max(dc.ram for dc in servers),
                'max_storage': max(dc.storage for dc in servers),
                'total_cpu': sum(dc.cpu for dc in servers),
                'total_ram': sum(dc.ram for dc in servers),
                'total_storage': sum(dc.storage for dc in servers),
                'avg_cpu': np.mean([dc.cpu for dc in servers]),
                'avg_ram': np.mean([dc.ram for dc in servers]),
                'avg_storage': np.mean([dc.storage for dc in servers]),
            }
        else:
            server_stats = {}
        
        # VNF statistics
        if vnf_specs:
            vnf_stats = {
                'max_cpu': max(v['cpu'] for v in vnf_specs.values()),
                'max_ram': max(v['ram'] for v in vnf_specs.values()),
                'max_storage': max(v['storage'] for v in vnf_specs.values()),
                'avg_cpu': np.mean([v['cpu'] for v in vnf_specs.values()]),
                'avg_ram': np.mean([v['ram'] for v in vnf_specs.values()]),
                'avg_storage': np.mean([v['storage'] for v in vnf_specs.values()]),
            }
        else:
            vnf_stats = {}
        
        # Topology statistics
        valid_bw = topology.bw_matrix[topology.bw_matrix > 0]
        valid_bw = valid_bw[valid_bw != np.inf]
        valid_delays = topology.delay_matrix[topology.delay_matrix > 0]
        valid_delays = valid_delays[valid_delays != np.inf]
        
        topology_stats = {
            'num_links': len(self.data["E"]),
            'max_bandwidth': np.max(valid_bw) if len(valid_bw) > 0 else 0,
            'avg_bandwidth': np.mean(valid_bw) if len(valid_bw) > 0 else 0,
            'max_delay': np.max(valid_delays) if len(valid_delays) > 0 else 0,
            'avg_delay': np.mean(valid_delays) if len(valid_delays) > 0 else 0,
        }
        
        return {
            'num_dcs': len(dc_list),
            'num_servers': len(servers),
            'num_vnf_types': len(vnf_specs),
            'num_requests': len(requests),
            'server_stats': server_stats,
            'vnf_stats': vnf_stats,
            'topology_stats': topology_stats
        }

