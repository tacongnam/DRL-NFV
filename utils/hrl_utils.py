from __future__ import annotations

import os
import copy
import numpy as np
from collections import OrderedDict
from typing import List, Optional


class LRUCache:
    def __init__(self, max_size: int = 1000):
        self.cache    = OrderedDict()
        self.max_size = max_size

    def get(self, key, default=None):
        if key not in self.cache:
            return default
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()


def snapshot_network(network) -> dict:
    node_snap = {}
    for nid, n in network.nodes.items():
        node_snap[nid] = {
            "used": {t: dict(v) for t, v in n.used.items()}
        }
    link_snap = []
    for lnk in network.links:
        link_snap.append({t: bw for t, bw in lnk.used.items()})
    return {"nodes": node_snap, "links": link_snap}


def restore_network(network, snap: dict):
    for nid, state in snap["nodes"].items():
        network.nodes[nid].used = state["used"]
    for lnk, used in zip(network.links, snap["links"]):
        lnk.used = used


def compute_revenue(req, weight_node: float = 1.0, weight_link: float = 1.0) -> float:
    import config
    duration = max(req.end_time - req.arrival_time, 1e-6)
    rev_node = weight_node * sum(
        sum(v.resource.get(k, 0.0) for k in config.RESOURCE_TYPE)
        for v in req.vnfs)
    rev_link = weight_link * req.bw
    return duration * (rev_node + rev_link)


def compute_cost(plan: dict, env, req,
                 weight_node: float = 1.0, weight_link: float = 1.0) -> float:
    import config
    duration  = max(req.end_time - req.arrival_time, 1e-6)
    node_cost = 0.0
    for v in plan.get("nodes", {}).values():
        dc_name  = v.get("dc", "")
        vnf_name = v.get("vnf_name", "")
        if dc_name in env.network.nodes and vnf_name in env.vnfs:
            node = env.network.nodes[dc_name]
            vnf  = env.vnfs[vnf_name]
            c    = node.get_cost(vnf)
            if c < float('inf'):
                node_cost += weight_node * c

    link_cost = 0.0
    for lp in plan.get("links", {}).values():
        path_len  = max(len(lp.get("path", [])) - 1, 0)
        link_cost += weight_link * req.bw * path_len

    return max(node_cost + link_cost * duration, 1e-6)