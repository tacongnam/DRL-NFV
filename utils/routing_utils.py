from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import config

class RoutingMixin:
    def __init__(self):
        self._bw_graph_cache:   Dict = {}
        self._path_len_cache:   Dict = {}
        self._routing_cache:    Dict = {}

    def clear_routing_cache(self):
        self._bw_graph_cache.clear()
        self._path_len_cache.clear()
        self._routing_cache.clear()

    def clear_episode_caches(self):
        self.clear_routing_cache()
        if hasattr(self, '_z_cache'):
            self._z_cache.clear()
        self._bw_graph_cache = {}

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
                G.add_edge(link.u.name, link.v.name,
                           weight=max(link.delay, 1e-6), delay=link.delay)
        self._bw_graph_cache[key] = G
        return G

    def _score_path(self, path: List[str], t_start: int, t_end: int,
                    bw: float, max_delay: float, max_hops: int) -> float:
        w_delay = getattr(config, 'ROUTING_DELAY_WEIGHT',    0.4)
        w_bw    = getattr(config, 'ROUTING_BW_WEIGHT',       0.3)
        w_mm1   = getattr(config, 'ROUTING_PRESSURE_WEIGHT', 0.2)
        w_hops  = getattr(config, 'ROUTING_HOP_WEIGHT',      0.1)
        if len(path) < 2:
            return 0.0
        edges = list(zip(path[:-1], path[1:]))
        total_delay, bw_pressures, mm1_pressures = 0.0, [], []
        for u, v in edges:
            link = next((lnk for lnk in self.env.network.links
                         if {lnk.u.name, lnk.v.name} == {u, v}), None)
            if link is None:
                continue
            total_delay += link.delay
            avail   = link.get_available_bandwidth(t_start, t_end)
            used_bw = link.cap - avail
            bw_pressures.append(self.link_pressure(avail - bw, link.cap))
            mm1_pressures.append(self._mm1_link_pressure(used_bw + bw, link.cap))
        delay_score = self._delay_norm(total_delay, max_delay)
        bw_score    = (sum(bw_pressures)  / len(bw_pressures))  if bw_pressures  else 0.0
        mm1_score   = (sum(mm1_pressures) / len(mm1_pressures)) if mm1_pressures else 0.0
        hop_score   = (len(path) - 1) / max(max_hops, 1)
        return w_delay * delay_score + w_bw * bw_score + w_mm1 * mm1_score + w_hops * hop_score

    def _yen_k_paths(self, G: nx.Graph, u: str, v: str, K: int) -> List[List[str]]:
        try:
            gen   = nx.shortest_simple_paths(G, u, v, weight="weight")
            paths = []
            for path in gen:
                paths.append(path)
                if len(paths) >= K:
                    break
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return []

    def get_routing(self, u: str, v: str, t_start: int, t_end: int,
                    bw: float, **kwargs) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]
        rkey = (u, v, t_start, t_end, round(bw, 2))
        if rkey in self._routing_cache:
            return self._routing_cache[rkey]
        K = getattr(config, 'ROUTING_K_PATHS', 3)
        G = self._bw_pruned_graph(t_start, t_end, bw)
        candidates = self._yen_k_paths(G, u, v, K)
        if not candidates:
            self._routing_cache[rkey] = None
            return None
        delays    = [link.delay for link in self.env.network.links if link.delay > 0]
        max_delay = max(delays, default=1.0) * len(candidates[0])
        max_hops  = max(len(p) - 1 for p in candidates)
        best_path, best_score = None, float('inf')
        for path in candidates:
            score = self._score_path(path, t_start, t_end, bw, max_delay, max_hops)
            if score < best_score:
                best_score, best_path = score, path
        self._routing_cache[rkey] = best_path
        return best_path

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