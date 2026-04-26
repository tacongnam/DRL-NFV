from __future__ import annotations
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple

import networkx as nx

class DStarLite:
    """
    D* Lite routing dùng latent space VGAE làm heuristic.
    h(u) = ||z_u - z_dest||_2
    Trọng số cạnh = link delay. Cạnh hết băng thông bị loại (weight = inf).
    """

    def __init__(self, network, latent_dim: int = 8):
        self.network    = network
        self.latent_dim = latent_dim

    def _heuristic(self, u: str, dest: str,
                   Z_map: Dict[str, np.ndarray]) -> float:
        if u not in Z_map or dest not in Z_map:
            return 0.0
        return float(np.linalg.norm(Z_map[u] - Z_map[dest]))

    def _edge_weight(self, u: str, v: str,
                     t_start: int, t_end: int, bw: float) -> float:
        for link in self.network.links:
            if {link.u.name, link.v.name} == {u, v}:
                if link.get_available_bandwidth(t_start, t_end) < bw:
                    return float('inf')
                return link.delay
        return float('inf')

    def _neighbors(self, node: str) -> List[str]:
        result = []
        for link in self.network.links:
            if link.u.name == node:
                result.append(link.v.name)
            elif link.v.name == node:
                result.append(link.u.name)
        return result

    def find_path(self, src: str, dest: str,
              t_start: int, t_end: int, bw: float,
              Z_t: np.ndarray,
              dc_mapping: List[str]) -> Optional[List[str]]:
        if src == dest:
            return [src]

        Z_map: Dict[str, np.ndarray] = {}
        for idx, dc_id in enumerate(dc_mapping):
            if idx < len(Z_t):
                Z_map[dc_id] = Z_t[idx]

        # Build adjacency map một lần, tránh duyệt links lặp lại
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for link in self.network.links:
            if link.get_available_bandwidth(t_start, t_end) < bw:
                continue
            u, v, d = link.u.name, link.v.name, link.delay
            adj.setdefault(u, []).append((v, d))
            adj.setdefault(v, []).append((u, d))

        INF = float('inf')
        g   = {}
        rhs = {src: 0.0}

        def h(u):
            return self._heuristic(u, dest, Z_map)

        def key(u):
            gu  = g.get(u, INF)
            ru  = rhs.get(u, INF)
            m   = min(gu, ru)
            return (m + h(u), m)

        heap = [(key(src), src)]
        parent: Dict[str, Optional[str]] = {src: None}
        visited = set()

        while heap:
            k, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            gu = g.get(u, INF)
            ru = rhs.get(u, INF)
            if ru < gu:
                g[u] = ru
            if u == dest:
                break
            for v, c in adj.get(u, []):
                new_rhs = g.get(u, INF) + c
                if new_rhs < rhs.get(v, INF):
                    rhs[v] = new_rhs
                    parent[v] = u
                    heapq.heappush(heap, (key(v), v))

        if dest not in g and rhs.get(dest, INF) == INF:
            return None

        path, cur = [], dest
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path if path and path[0] == src else None