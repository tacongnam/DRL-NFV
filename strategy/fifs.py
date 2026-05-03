from typing import Dict, List, Optional

from env.env import Strategy
from env.request import SFC
from strategy.routing_utils import RoutingMixin
import config

class GreedyFIFS(Strategy, RoutingMixin):
    """FIFO + cost-based DC selection. Uses RoutingMixin for routing."""
    
    def __init__(self, env):
        super().__init__(env)
        RoutingMixin.__init__(self)
        self.name = "GreedyFIFS"

    def get_placement(self, sfc: SFC, current_time: float) -> Optional[Dict]:
        self._graph_cache.clear()
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

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
                if self.env._check_can_deploy_vnf(dc_node, vnf, t_start, t_end):
                    valid_dcs.append((dc_node.get_cost(vnf), str(dc_name)))

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