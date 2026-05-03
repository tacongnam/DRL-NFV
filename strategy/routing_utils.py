from typing import Dict, List, Optional
import networkx as nx
from env.env import Strategy
from env.request import SFC


class RoutingMixin:    
    def __init__(self):
        self._graph_cache: Dict = {}
    
    def clear_routing_cache(self):
        self._graph_cache.clear()
    
    def _bw_pruned_graph(self, t_start: int, t_end: int, bw: float) -> nx.Graph:
        key = (t_start, t_end, round(bw, 2))
        if key in self._graph_cache:
            return self._graph_cache[key]
        
        G = nx.Graph()
        for nid in self.env.network.nodes:
            G.add_node(nid)
        
        for link in self.env.network.links:
            if link.get_available_bandwidth(t_start, t_end) >= bw:
                G.add_edge(link.u.name, link.v.name, delay=link.delay)
        
        self._graph_cache[key] = G
        return G
    
    def get_routing(self, u: str, v: str, t_start: int, t_end: int, 
                    bw: float) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]
        
        G = self._bw_pruned_graph(t_start, t_end, bw)
        
        if u not in G or v not in G:
            return None
        
        try:
            return nx.shortest_path(G, u, v, weight='delay')
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None
