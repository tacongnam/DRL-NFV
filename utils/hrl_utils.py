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