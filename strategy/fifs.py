from typing import Dict, List, Optional
import networkx as nx

from env.env import Strategy
from env.request import SFC
from env.network import Link
import config

class GreedyFIFS(Strategy):
    """Greedy First-In-First-Served: chọn DC có cost thấp nhất."""

    def __init__(self, env):
        super().__init__(env)
        self.name = "GreedyFIFS"

    def get_routing(self, u: str, v: str, t_start: int, t_end: int, bw: float) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return[u]

        G = self.env.network.to_graph()
        if u not in G or v not in G:
            return None

        edges_to_remove =[]
        for a, b in G.edges():
            link_obj = next(
                (l for l in self.env.network.links if {l.u.name, l.v.name} == {a, b}),
                None
            )
            if not link_obj or link_obj.get_available_bandwidth(t_start, t_end) < bw:
                edges_to_remove.append((a, b))

        G.remove_edges_from(edges_to_remove)

        try:
            return nx.shortest_path(G, u, v, weight='delay')
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None

    def get_placement(self, sfc: SFC, current_time: float) -> Optional[Dict]:
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

        node_placements, vnf_timeslots = [], []
        link_paths,      link_timeslots = [], []
        prev_dc = sfc.request.start_node

        for vnf_idx, vnf in enumerate(sfc.request.vnfs):
            candidate_dcs = vnf.get_dcs()
            if '-1' in candidate_dcs:
                candidate_dcs = [n.name for n in self.env.network.get_dc_node()]

            valid_dcs = []
            for dc_name in candidate_dcs:
                dc_node = self.env.network.nodes.get(str(dc_name))
                if dc_node is None:
                    continue
                if self.env._check_can_deploy_vnf(dc_node, vnf, t_start, t_end):
                    valid_dcs.append((dc_node.get_cost(vnf), str(dc_name)))

            if not valid_dcs:
                return None

            valid_dcs.sort(key=lambda x: x[0])
            chosen_dc = None
            path = None
            for score, dc_candidate in valid_dcs:
                p = self.get_routing(prev_dc, dc_candidate, t_start, t_end, sfc.request.bw)
                if p is not None:
                    chosen_dc = dc_candidate
                    path = p
                    break  # Tìm thấy DC và đường đi hợp lệ -> Chốt

            if chosen_dc is None or path is None:
                return None

            node_placements.append(chosen_dc)
            vnf_timeslots.append((t_start, t_end))
            link_paths.append(path)
            link_timeslots.append((t_start, t_end))
            prev_dc = chosen_dc

        # Segment cuối: VNF cuối → end_node
        final_path = self.get_routing(prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None
        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))

        return self.build_placement_plan(node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)