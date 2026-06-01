import math
import numpy as np
from typing import Optional, List
import config

def resolve_request_limit(total_requests: int, request_pct: int = 0) -> Optional[int]:
    if request_pct is None or request_pct <= 0:
        return None
    
    limit = max(1, math.ceil(total_requests * request_pct / 100.0))
    return limit

def sample_requests(req_rows: list, request_pct: int = 0) -> list:
    req_limit = resolve_request_limit(len(req_rows), request_pct=request_pct)
    if req_limit is None or req_limit <= 0 or len(req_rows) <= req_limit:
        return req_rows
    idxs = np.linspace(0, len(req_rows) - 1, num=req_limit, dtype=int)
    return [req_rows[i] for i in idxs]

def compute_epsilon(progress: float) -> float:
    pw = config.EPSILON_WARMUP
    if progress < pw:
        return config.EPSILON_MAX
    t = (progress - pw) / max(1.0 - pw, 1e-6)
    return config.EPSILON_MIN + 0.5 * (config.EPSILON_MAX - config.EPSILON_MIN) * (
        1.0 + math.cos(math.pi * t))

def estimate_max_cost(env, sfc) -> float:
    return max(1.0, sum(
        max((n.get_cost(v) for n in env.network.nodes.values()
             if n.type == config.NODE_DC and n.cost is not None
             and n.get_cost(v) < float('inf')), default=0.0)
        for v in sfc.request.vnfs))

def edf_sort_key(sfc) -> float:
    return sfc.request.end_time
