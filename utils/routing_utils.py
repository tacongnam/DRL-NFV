from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
import networkx as nx
import config


class RoutingMixin:
    def __init__(self):
        self._bw_graph_cache:  Dict = {}
        self._path_len_cache:  Dict = {}
        self._routing_cache:   Dict = {}

    def clear_routing_cache(self):
        self._bw_graph_cache.clear()
        self._path_len_cache.clear()
        self._routing_cache.clear()

    def clear_episode_caches(self):
        self.clear_routing_cache()
        if hasattr(self, '_z_cache'):
            self._z_cache.clear()

    # ── Static pressure helpers (giữ để dùng ngoài routing) ──

    @staticmethod
    def link_pressure(remaining_bw: float, capacity: float) -> float:
        if capacity <= 0:
            return 1.0
        return math.exp(-max(remaining_bw, 0.0) / capacity)

    @staticmethod
    def _mm1_link_pressure(used_bw: float, capacity: float) -> float:
        if capacity <= 0:
            return 1.0
        load  = min(used_bw / capacity, 0.999)
        omega = max(1.0 - load, 0.001)
        return min((load / omega) / 20.0, 1.0)

    @staticmethod
    def _delay_norm(delay: float, max_delay: float) -> float:
        return delay / max(max_delay, 1e-6)

    # ── Graph building ────────────────────────────────────────

    def _bw_pruned_graph(self, t_start: int, t_end: int, bw: float) -> nx.Graph:
        """
        Xây graph với edge weight = composite_weight từ Link.
        Dijkstra trên graph này cho path tối ưu trực tiếp —
        không cần Yen's K-paths + score riêng.
        """
        key = (t_start, t_end, round(bw, 2))
        if key in self._bw_graph_cache:
            return self._bw_graph_cache[key]

        all_delays = [lnk.delay for lnk in self.env.network.links if lnk.delay > 0]
        ref_delay  = max(all_delays, default=1.0)

        w_delay = getattr(config, 'ROUTING_DELAY_WEIGHT',    0.4)
        w_bw    = getattr(config, 'ROUTING_BW_WEIGHT',       0.3)
        w_mm1   = getattr(config, 'ROUTING_PRESSURE_WEIGHT', 0.2)
        w_hops  = getattr(config, 'ROUTING_HOP_WEIGHT',      0.1)

        G = nx.Graph()
        for nid in self.env.network.nodes:
            G.add_node(nid)
        for lnk in self.env.network.links:
            if lnk.get_available_bandwidth(t_start, t_end) < bw:
                continue
            G.add_edge(
                lnk.u.name, lnk.v.name,
                weight=lnk.composite_weight(
                    t_start, t_end, bw,
                    ref_delay, w_delay, w_bw, w_mm1, w_hops),
                delay=lnk.delay)

        self._bw_graph_cache[key] = G
        return G

    # ── Routing ───────────────────────────────────────────────

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

    # ── Pressure helpers ──────────────────────────────────────

    def avg_path_pressure(self, sfc, t: float, path: List[str] = None) -> float:
        t_start = self.env._get_timeslot(t)
        t_end   = self.env._get_timeslot(sfc.request.end_time)
        if path is not None and len(path) > 1:
            path_edges   = set(zip(path[:-1], path[1:]))
            target_links = [lnk for lnk in self.env.network.links
                            if (lnk.u.name, lnk.v.name) in path_edges
                            or (lnk.v.name, lnk.u.name) in path_edges]
        else:
            target_links = self.env.network.links
        pressures = []
        for link in target_links:
            avail   = link.get_available_bandwidth(t_start, t_end)
            used_bw = link.cap - avail
            pressures.append(self._mm1_link_pressure(used_bw + sfc.request.bw, link.cap))
        return float(sum(pressures) / len(pressures)) if pressures else 0.0

    def mean_candidate_pressure(self, valid_indices: List[int], dc_mapping: List[str],
                                 vnf, t_start: int, t_end: int) -> float:
        pressures = []
        for idx in valid_indices:
            dc_id = dc_mapping[idx]
            node  = self.env.network.nodes[dc_id]
            res   = node.get_min_available_resource(t_start, t_end)
            cap   = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
            from models.placer import PressureNode
            pressures.append(PressureNode.compute(res, vnf.resource, cap))
        return float(sum(pressures) / len(pressures)) if pressures else 0.0