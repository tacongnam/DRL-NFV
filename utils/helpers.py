import math
import numpy as np
from typing import Optional, List


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
