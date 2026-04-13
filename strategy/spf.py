from typing import Dict, List, Optional
import networkx as nx

from env.env import Strategy
from env.request import SFC
import config

class ShortestPathFirst(Strategy):
    def __init__(self, env):
        super().__init__(env)
        self.name = "ShortestPathFirst"
        self._graph_cache: dict = {}
        self._path_cache: dict = {}

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

    def get_routing(self, u: str, v: str, t_start: int, t_end: int, bw: float) -> Optional[List[str]]:
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

    def _path_delay(self, path: List[str]) -> float:
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            link = next((l for l in self.env.network.links
                         if {l.u.name, l.v.name} == {u, v}), None)
            if link:
                total += link.delay
        return total

    def _all_pairs_delay(self, t_start: int, t_end: int, bw: float) -> dict:
        key = (t_start, t_end, round(bw, 2))
        if key in self._path_cache:
            return self._path_cache[key]
        G = self._bw_pruned_graph(t_start, t_end, bw)
        try:
            result = dict(nx.shortest_path_length(G, weight='delay'))
        except Exception:
            result = {}
        self._path_cache[key] = result
        return result

    def get_placement(self, sfc: SFC, current_time: float) -> Optional[Dict]:
        self._graph_cache.clear()
        self._path_cache.clear()
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

        all_delays = self._all_pairs_delay(t_start, t_end, sfc.request.bw)

        node_placements, vnf_timeslots = [], []
        link_paths,      link_timeslots = [], []
        prev_dc = sfc.request.start_node

        for vnf in sfc.request.vnfs:
            candidate_dcs = vnf.get_dcs()
            if '-1' in candidate_dcs:
                candidate_dcs = [n.name for n in self.env.network.get_dc_node()]

            valid_dcs = []
            for dc_name in candidate_dcs:
                dc_node = self.env.network.nodes.get(str(dc_name))
                if dc_node is None:
                    continue
                if not self.env._check_can_deploy_vnf(dc_node, vnf, t_start, t_end):
                    continue
                delay = all_delays.get(prev_dc, {}).get(str(dc_name), float('inf'))
                valid_dcs.append((delay, str(dc_name)))

            if not valid_dcs:
                return None

            valid_dcs.sort(key=lambda x: x[0])
            chosen_dc, path = None, None
            for _, dc_candidate in valid_dcs:
                p = self.get_routing(prev_dc, dc_candidate, t_start, t_end, sfc.request.bw)
                if p is not None:
                    chosen_dc, path = dc_candidate, p
                    break

            if chosen_dc is None:
                return None

            node_placements.append(chosen_dc)
            vnf_timeslots.append((t_start, t_end))
            link_paths.append(path)
            link_timeslots.append((t_start, t_end))
            prev_dc = chosen_dc

        final_path = self.get_routing(prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None
        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))

        return self.build_placement_plan(node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)