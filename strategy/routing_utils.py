import math
from typing import Dict, List, Optional
import networkx as nx
import config

class RoutingMixin:
    def __init__(self):
        self._bw_graph_cache: Dict = {}
        self._path_len_cache: Dict = {}
        self._routing_cache: Dict = {}

    def clear_routing_cache(self):
        self._bw_graph_cache.clear()
        self._path_len_cache.clear()
        self._routing_cache.clear()

    @staticmethod
    def link_pressure(remaining_bw: float, capacity: float) -> float:
        if capacity <= 0:
            return 1.0
        return math.exp(-max(remaining_bw, 0.0) / capacity)

    def _bw_pruned_graph(self, t_start: int, t_end: int, bw: float) -> nx.Graph:
        key = (t_start, t_end, round(bw, 2))
        if key in self._bw_graph_cache:
            return self._bw_graph_cache[key]
        G = nx.Graph()
        for nid in self.env.network.nodes:
            G.add_node(nid)
        for link in self.env.network.links:
            avail = link.get_available_bandwidth(t_start, t_end)
            if avail >= bw:
                pressure = self.link_pressure(avail - bw, link.cap)
                w = link.delay + config.ROUTING_PRESSURE_WEIGHT * pressure
                G.add_edge(link.u.name, link.v.name, weight=w, delay=link.delay)
        self._bw_graph_cache[key] = G
        return G

    def _get_all_path_lengths(self, t_start: int, t_end: int, bw: float) -> dict:
        key = (t_start, t_end, round(bw, 2))
        if key in self._path_len_cache:
            return self._path_len_cache[key]
        G = self._bw_pruned_graph(t_start, t_end, bw)
        try:
            all_len = dict(nx.shortest_path_length(G, weight="weight"))
        except Exception:
            all_len = {}
        self._path_len_cache[key] = all_len
        return all_len

    def get_routing(self, u: str, v: str, t_start: int, t_end: int,
                    bw: float, **kwargs) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]
        rkey = (u, v, t_start, t_end, round(bw, 2))
        if rkey in self._routing_cache:
            return self._routing_cache[rkey]
        G = self._bw_pruned_graph(t_start, t_end, bw)
        try:
            path = nx.shortest_path(G, u, v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            path = None
        self._routing_cache[rkey] = path
        return path