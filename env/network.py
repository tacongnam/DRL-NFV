import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import config
from env.vnf import VNF


class Node:
    def __init__(self, name: str, type: int, delay: float = 0.0,
                 cap: dict = None, cost: dict = None,
                 used: dict = None, VNFs: set = None):
        self.name  = name
        if type not in [config.NODE_DC, config.NODE_SWITCH]:
            print(f"Node {name}: type phải là 0 (DC) hoặc 1 (Switch), nhận {type}")
            return
        self.type  = type
        self.delay = delay
        self.cap   = cap
        self.cost  = cost
        self.links = []
        self.vnfs  = set() if VNFs is None else VNFs

        if used is None:
            # Timeslot 0 khởi tạo với usage = 0
            self.used = {0: {k: 0.0 for k in config.RESOURCE_TYPE}}
        else:
            self.used = used

    def __repr__(self):
        t = "Switch" if self.type == config.NODE_SWITCH else "DC"
        return f"Node {self.name} ({t})"

    def check_violated(self, T: int, resource: dict = None) -> bool:
        if self.type == config.NODE_SWITCH or self.cap is None:
            return False
        if resource is None:
            resource = {k: 0.0 for k in config.RESOURCE_TYPE}
        if T in self.used:
            used_at_T = self.used[T]
        else:
            prev_t = max((t for t in self.used if t < T), default=None)
            used_at_T = self.used[prev_t] if prev_t is not None else {k: 0.0 for k in config.RESOURCE_TYPE}
        return any(used_at_T[k] + resource[k] > self.cap[k] for k in config.RESOURCE_TYPE)

    def get_min_available_resource(self, t_start: int, t_end: int) -> dict:
        min_res = {k: self.cap[k] for k in config.RESOURCE_TYPE}
        relevant = [t for t in self.used if t_start <= t <= t_end]
        if not relevant:
            prev_t = max((t for t in self.used if t < t_start), default=None)
            if prev_t is not None:
                for k in config.RESOURCE_TYPE:
                    min_res[k] = self.cap[k] - self.used[prev_t][k]
            return min_res
        for t in relevant:
            for k in config.RESOURCE_TYPE:
                min_res[k] = min(min_res[k], self.cap[k] - self.used[t][k])
        return min_res

    def use(self, resource: dict, start_T: int, end_T: int):
        # Khi T không có trong used, lấy giá trị từ key liền trước thay vì T-1 (T-1 có thể cũng không tồn tại)
        for T in range(start_T, end_T):
            if T not in self.used:
                prev_t = max((t for t in self.used if t < T), default=None)
                prev = self.used[prev_t] if prev_t is not None else {k: 0.0 for k in config.RESOURCE_TYPE}
                self.used[T] = {k: prev[k] for k in config.RESOURCE_TYPE}
            for k in config.RESOURCE_TYPE:
                self.used[T][k] += resource[k]

    def get_cost(self, vnf: VNF) -> float:
        if self.type == config.NODE_SWITCH or self.cost is None:
            return float('inf')
        return sum(vnf.resource[k] * self.cost[k] for k in config.RESOURCE_TYPE)

    def get_state(self, T: int) -> dict:
        if T in self.used:
            used_T = self.used[T]
        else:
            prev_t = max((t for t in self.used if t < T), default=None)
            used_T = self.used[prev_t] if prev_t is not None else {k: 0.0 for k in config.RESOURCE_TYPE}
        return {k: self.cap[k] - used_T[k] for k in config.RESOURCE_TYPE}

    def exist_vnf(self, vnf_id) -> bool:
        return self.type == config.NODE_DC and vnf_id in self.vnfs

    def install_vnf(self, vnf_id) -> bool:
        if not self.exist_vnf(vnf_id):
            self.vnfs.add(vnf_id)
            return True
        return False

    def remove_vnf(self, vnf_id) -> bool:
        if self.exist_vnf(vnf_id):
            self.vnfs.remove(vnf_id)
            return True
        return False


class Link:
    def __init__(self, u: Node, v: Node, bandwidth_capacity: float,
                 link_delay: float, used: dict = None):
        self.u     = u
        self.v     = v
        self.cap   = bandwidth_capacity
        self.delay = link_delay
        self.used  = {0: 0.0} if used is None else used

    def next(self, node):
        if node is self.v:   return self.u
        if node is self.u:   return self.v
        return None

    def check_violated(self, T: int, bandwidth: float = 0.0) -> bool:
        if T in self.used:
            used_T = self.used[T]
        else:
            prev_t = max((t for t in self.used if t < T), default=None)
            used_T = self.used[prev_t] if prev_t is not None else 0.0
        return used_T + bandwidth > self.cap

    def use(self, bandwidth: float, start_T: int, end_T: int):
        for T in range(start_T, end_T):
            if T not in self.used:
                prev_t = max((t for t in self.used if t < T), default=None)
                self.used[T] = self.used[prev_t] if prev_t is not None else 0.0
            self.used[T] += bandwidth

    def get_available_bandwidth(self, start_T: int, end_T: int) -> float:
        bw = self.cap
        relevant = [t for t in self.used if start_T <= t < end_T]
        if not relevant:
            prev_t = max((t for t in self.used if t < start_T), default=None)
            if prev_t is not None:
                bw = min(bw, self.cap - self.used[prev_t])
            return bw
        for t in relevant:
            bw = min(bw, self.cap - self.used[t])
        return bw

    def get_state(self, T: int) -> float:
        if T in self.used:
            used_T = self.used[T]
        else:
            prev_t = max((t for t in self.used if t < T), default=None)
            used_T = self.used[prev_t] if prev_t is not None else 0.0
        return self.cap - used_T    

class Network:
    def __init__(self):
        self.nodes: dict = {}
        self.links: list = []

    def get_switch_node(self):
        return [n for n in self.nodes.values() if n.type == config.NODE_SWITCH]

    def get_dc_node(self):
        return [n for n in self.nodes.values() if n.type == config.NODE_DC]

    def add_switch_node(self, name: str):
        self.nodes[name] = Node(name, config.NODE_SWITCH)

    def add_dc_node(self, name: str, delay: float, capacity: dict,
                    cost: dict = None, used: dict = None, VNFs: set = None):
        self.nodes[name] = Node(name, config.NODE_DC, delay, capacity, cost, used, VNFs)

    def add_link(self, nu: str, nv: str, bandwidth_capacity: float,
                 delay: float, used: dict = None):
        u    = self.nodes[nu]
        v    = self.nodes[nv]
        link = Link(u, v, bandwidth_capacity, delay, used)
        self.links.append(link)
        u.links.append(link)
        v.links.append(link)

    def add_nodes(self, node_list):
        for node in node_list:
            if node.type == config.NODE_DC:
                self.add_dc_node(node.name, node.delay, node.cap,
                                 node.cost, node.used, node.vnfs)
            else:
                self.add_switch_node(node.name)

    def add_links(self, link_list):
        for link in link_list:
            self.add_link(link.u.name, link.v.name,
                          link.cap, link.delay, link.used)

    def check_violated_nodes(self, T: int) -> int:
        return sum(n.check_violated(T) for n in self.get_dc_node())

    def check_violated_links(self, T: int) -> int:
        return sum(l.check_violated(T) for l in self.links)

    def to_graph(self) -> nx.Graph:
        G = nx.Graph()
        for node in self.nodes.values():
            G.add_node(node.name, type=node.type, cap=node.cap,
                       used=node.used, VNFs=node.vnfs, links=node.links)
        for link in self.links:
            G.add_edge(link.u.name, link.v.name,
                       cap=link.cap, used=link.used, delay=link.delay)
        return G

    def visualize(self, pos_path=None, info=False, topo=False,
                  out_path=None, path_p=None):
        G = self.to_graph()
        if pos_path is None:
            pos = nx.spring_layout(G)
        else:
            df  = pd.read_csv(pos_path, skiprows=1)
            arr = df.to_numpy()
            arr[:, 1] = -arr[:, 1]
            pos = {str(i): c for i, c in enumerate(arr)}

        colors = (['blue' if G.nodes[n]['type'] == config.NODE_SWITCH else 'red'
                   for n in G.nodes] if topo else ['black'])
        nx.draw(G, pos, node_size=50, node_color=colors,
                with_labels=True, font_size=5, font_color='white')

        if info:
            pos_off = {n: (c[0], c[1] + 10) for n, c in pos.items()}
            nx.draw_networkx_labels(
                G, pos_off,
                labels={n: f"{G.nodes[n]['used']}/{G.nodes[n]['cap']}" for n in G.nodes},
                font_size=5)
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels={(u, v): f"{G[u][v]['used']}/{G[u][v]['cap']}"
                             for u, v in G.edges()},
                font_size=5)

        if path_p:
            edges_p = list(zip(path_p, path_p[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=path_p,
                                   node_color='grey', node_size=80)
            nx.draw_networkx_edges(G, pos, edgelist=edges_p,
                                   edge_color='blue', width=2)

        if out_path:
            plt.savefig(out_path)
        plt.show()

    def visualize_dynamic(self, timeslots, pause_time=1.0):
        G   = self.to_graph()
        pos = nx.spring_layout(G)
        plt.ion()
        for t in timeslots:
            plt.clf()
            cols = ['blue' if G.nodes[n]['type'] == config.NODE_SWITCH else 'red'
                    for n in G.nodes]
            nx.draw(G, pos, node_color=cols, with_labels=True, node_size=500)
            lbls = {
                n: f"T={t}\n"
                   f"{G.nodes[n]['used'].get(t, {'cpu': 0.0})['cpu']:.1f}"
                   f"/{G.nodes[n]['cap']['cpu']:.1f}"
                for n in G.nodes if G.nodes[n]['type'] == config.NODE_DC
            }
            nx.draw_networkx_labels(
                G, {n: (c[0], c[1] + 0.15) for n, c in pos.items()}, lbls)
            plt.pause(pause_time)
        plt.ioff()
        plt.show()