import json
import sys
import numpy as np
sys.path.append('..')

from DRL.core.dc import DataCenter
from DRL.core.topology import TopologyManager

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
                               cost_c=cost_c, cost_h=cost_h, cost_r=cost_r)
                dc_list.append(dc)
            else:
                dc = DataCenter(dc_id)
                dc_list.append(dc)
        return dc_list
    
    def get_E(self, num_dcs):
        # Initialize matrices
        delay_matrix = np.zeros((num_dcs, num_dcs))
        bw_matrix = np.zeros((num_dcs, num_dcs))
        
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
    
    # def get_F(self):
    #     vnf_list = []
    #     for i in range(len(self.data["F"])):
    #         c_f = self.data["F"][i]["c_f"]
    #         r_f = self.data["F"][i]["r_f"]
    #         h_f = self.data["F"][i]["h_f"]
    #         d_f = {}
    #         for key, value in self.data["F"][i]["d_f"].items():
    #             d_f[int(key)] = value
    #         vnf = VNF(i, c_f, r_f, h_f, d_f)
    #         vnf_list.append(vnf)
    #     return vnf_list
    
    # def get_R(self):
    #     r_list = []
    #     name = 0
    #     for r in self.data["R"]:
    #         T = r["T"] + 1
    #         st_r = r["st_r"]
    #         d_r = r["d_r"]
    #         F_r = r["F_r"]
    #         b_r = r["b_r"]
    #         d_max = int(r["d_max"])+T
    #         request = Request(r["d_max"], name, T, d_max, st_r, d_r, F_r, b_r)
    #         r_list.append(request)
    #         name = name + 1
    #     return r_list
    
    # def get_info_network(self):
    #     server_list = self.get_V()
    #     link_list = self.get_E()
    #     vnf_list = self.get_F()
    #     print(len(server_list), len(link_list), len(vnf_list))
    #     ram_max_server, cpu_max_server, mem_max_server, sum_ram_server, sum_cpu_server, sum_mem_server = get_info_server(server_list)
    #     ram_max_vnf, cpu_max_vnf, mem_max_vnf = get_info_vnf(vnf_list)
    #     max_bandwidth = get_info_link(link_list)
    #     return ram_max_server, cpu_max_server, mem_max_server, sum_ram_server, sum_cpu_server, sum_mem_server, ram_max_vnf, cpu_max_vnf, mem_max_vnf, max_bandwidth

