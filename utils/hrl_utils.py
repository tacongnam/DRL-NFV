from __future__ import annotations

import os
import numpy as np
from collections import OrderedDict
from typing import List, Optional

class LRUCache:
    """Simple LRU (Least Recently Used) cache with size limit."""
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
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
    return {
        "nodes": {nid: {t: dict(v) for t, v in n.used.items()}
                  for nid, n in network.nodes.items()},
        "links": [{t: bw for t, bw in lnk.used.items()} for lnk in network.links],
    }

def restore_network(network, snap: dict):
    for nid, used in snap["nodes"].items():
        network.nodes[nid].used = used
    for lnk, used in zip(network.links, snap["links"]):
        lnk.used = used

def resolve_npy_path(path: str, default_filename: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, default_filename)
    if path.endswith(".weights.h5"):
        return path[: -len(".weights.h5")] + ".npy"
    if not path.endswith(".npy"):
        return path + ".npy"
    return path

def get_next_time(pending: List, current_t: float) -> float:
    return min((s.request.arrival_time for s in pending), default=current_t) if pending else current_t

def extract_node_plan_map(plan: dict) -> dict:
    result = {}
    for vnf_key, vplan in plan.get("nodes", {}).items():
        idx = int(vnf_key.split("_")[0])
        result[idx] = vplan
    return result
