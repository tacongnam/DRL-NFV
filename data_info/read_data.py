import json
import networkx as nx
import config


class Read_data:
    """Load JSON data (V, E, F, R format) and build physical network graph"""

    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        self.graph = None
        self._build_graph()

    # ------------------------------------------------------------------
    # Physical network
    # ------------------------------------------------------------------
    def _build_graph(self):
        self.graph = nx.Graph()

        # ---------- Nodes ----------
        for node_id, node in self.data["V"].items():
            node_id = int(node_id)

            if node.get("server", False):
                self.graph.add_node(
                    node_id,
                    server=True,
                    cpu=node.get("c_v", config.DC_CPU_CYCLES),
                    ram=node.get("r_v", config.DC_RAM),
                    storage=node.get("h_v", config.DC_STORAGE),
                    delay=node.get("d_v", 0.0),
                    cost_c=node.get("cost_c", 1.0),
                    cost_r=node.get("cost_r", 1.0),
                    cost_h=node.get("cost_h", 1.0),
                )
            else:
                self.graph.add_node(node_id, server=False)

        # ---------- Links ----------
        for link in self.data["E"]:
            u = int(link["u"])
            v = int(link["v"])
            bw = link.get("b_l", config.LINK_BW_CAPACITY)
            delay = link.get("d_l", 1.0)

            self.graph.add_edge(
                u,
                v,
                bw=bw,
                capacity=bw,
                delay=delay
            )

    def get_G(self):
        return self.graph

    # ------------------------------------------------------------------
    # Data centers
    # ------------------------------------------------------------------
    def get_V(self):
        from core.dc import DataCenter

        dcs = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("server", False):
                dc = DataCenter(
                    dc_id=node_id,
                    cpu=attrs["cpu"],
                    ram=attrs["ram"],
                    storage=attrs["storage"],
                    delay=attrs["delay"],
                    cost_c=attrs["cost_c"],
                    cost_h=attrs["cost_h"],
                    cost_r=attrs["cost_r"],
                    is_server=True,
                )
            else:
                dc = DataCenter(dc_id=node_id, is_server=False)

            dcs.append(dc)

        return dcs

    # ------------------------------------------------------------------
    # VNF types
    # ------------------------------------------------------------------
    def get_F(self):
        """
        Returns:
            dict[vnf_id] = {
                cpu, ram, storage, startup_time{dc_id: delay}
            }
        """
        vnf_specs = {}

        for idx, vnf in enumerate(self.data["F"]):
            vnf_specs[idx] = {
                "cpu": vnf.get("c_f", 1.0),
                "ram": vnf.get("r_f", 1.0),
                "storage": vnf.get("h_f", 1.0),
                "startup_time": {
                    int(dc_id): delay
                    for dc_id, delay in vnf.get("d_f", {}).items()
                },
            }

        return vnf_specs

    # ------------------------------------------------------------------
    # SFC requests
    # ------------------------------------------------------------------
    def get_R(self):
        """
        Returns list of request dicts
        """
        requests = []

        for idx, req in enumerate(self.data["R"]):
            requests.append({
                "id": idx,
                "arrival_time": req.get("T", 0),
                "source": req["st_r"],
                "destination": req["d_r"],
                "vnf_chain": req["F_r"],
                "bandwidth": req.get("b_r", 1.0),
                "max_delay": req.get("d_max", 100.0),
            })

        return requests

    # ------------------------------------------------------------------
    def get_num_servers(self):
        return sum(
            1 for _, attrs in self.graph.nodes(data=True)
            if attrs.get("server", False)
        )
