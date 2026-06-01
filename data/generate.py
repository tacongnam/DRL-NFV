"""
Feasibility-Guaranteed Synthetic Data Generator for NFV VNF Placement.

Đảm bảo acceptance ratio tối thiểu bằng cách:
  1. Kiểm tra connectivity trước khi tạo request
  2. Tính toán delay_max dựa trên path length thực tế (không dual-use)
  3. Kiểm tra tổng VNF resource <= capacity khả dụng
  4. Đảm bảo BW path feasible từ src đến dst
  5. Tham số difficulty kiểm soát resource utilization target, không phải lifetime

Usage:
    python data/generate.py --topology nsf --distribution rural --difficulty easy
    python data/generate.py --topology cogent --distribution centers --difficulty hard
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Topology definitions
# ---------------------------------------------------------------------------
TOPOLOGIES = {
    "nsf": {
        "nodes": list(range(14)),
        "links": [
            (0,1),(0,3),(0,4),(1,2),(1,7),(2,3),(2,10),(3,12),
            (4,5),(4,8),(5,6),(5,9),(5,12),(6,9),(6,11),(7,11),
            (7,13),(8,10),(8,13),(9,13),(10,12),(11,13)
        ],
    },
    "conus": {
        "nodes": list(range(75)),
        "links": None,   # generated
    },
    "cogent": {
        "nodes": list(range(197)),
        "links": None,   # generated
    }
}

# Difficulty controls resource utilization target per request
DIFFICULTY_CFG = {
    #            util_target  bw_choices        lifetime_multiplier  latency_slack
    "easy":  dict(util=0.15,  bw=[1.0],         lt_mul=15.0,         lat_slack=3.0),
    "normal":dict(util=0.25,  bw=[1.0, 2.0],    lt_mul=10.0,         lat_slack=2.0),
    "hard":  dict(util=0.40,  bw=[2.0, 5.0],    lt_mul=6.0,          lat_slack=1.5),
}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def generate_grid_links(n_nodes: int, extra_link_ratio: float = 0.5):
    """Connected graph với spanning chain + extra random links."""
    links = set()
    perm  = list(range(n_nodes))
    random.shuffle(perm)
    for i in range(n_nodes - 1):
        links.add((min(perm[i], perm[i+1]), max(perm[i], perm[i+1])))
    all_pairs = [
        (i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)
        if (i, j) not in links
    ]
    n_extra = int(n_nodes * extra_link_ratio)
    for i, j in random.sample(all_pairs, min(n_extra, len(all_pairs))):
        links.add((i, j))
    return sorted(links)


def build_nx_graph(links) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(links)
    return G


def get_node_degrees(G: nx.Graph):
    """Trả về list node sắp xếp theo degree giảm dần."""
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# Node / link generation
# ---------------------------------------------------------------------------

def select_server_nodes(G: nx.Graph, distribution: str, fraction: float = 0.3):
    """
    Chọn server nodes dựa trên degree thực tế của graph.
    Đảm bảo server nodes có connectivity tốt.
    """
    nodes_by_degree = get_node_degrees(G)
    n = len(nodes_by_degree)

    if distribution == "uniform":
        return random.sample(nodes_by_degree, max(2, int(n * fraction)))
    elif distribution == "rural":
        # Low-degree nodes — ít kết nối hơn, giống mạng nông thôn
        low_degree = nodes_by_degree[int(n * 0.5):]   # bottom 50% degree
        return random.sample(low_degree, max(2, int(len(low_degree) * 0.6)))
    elif distribution == "urban":
        # High-degree nodes — hub nodes
        high_degree = nodes_by_degree[:max(2, int(n * 0.4))]
        return random.sample(high_degree, max(2, int(len(high_degree) * 0.7)))
    elif distribution == "centers":
        # Top degree nodes — backbone DCs
        return nodes_by_degree[:max(2, int(n * 0.3))]
    return random.sample(nodes_by_degree, max(2, int(n * fraction)))


def generate_nodes(all_nodes, server_nodes_set: set, scale: int,
                   difficulty: str) -> dict:
    """
    Tạo node với capacity đảm bảo có thể chứa requests.
    scale: resource multiplier (50 = standard, 100 = double)
    """
    cfg        = DIFFICULTY_CFG[difficulty]
    util_tgt   = cfg["util"]
    scale_f    = scale / 50.0
    V          = {}

    # Base capacity — đủ để chứa nhiều VNFs đồng thời
    # Với util_target=0.15 và 50 requests, capacity cần ~7-8× tổng VNF demand
    base_cap = {
        "cpu": int(200 * scale_f),
        "ram": int(128 * scale_f),
        "mem": int(200 * scale_f),
    }

    for node_id in all_nodes:
        name = f"v{node_id}"
        if node_id in server_nodes_set:
            # Thêm variance nhỏ để không đồng đều hoàn toàn
            variance = random.uniform(0.8, 1.2)
            V[name] = {
                "server": True,
                "c_v": max(10, int(base_cap["cpu"] * variance)),
                "r_v": max(8,  int(base_cap["ram"] * variance)),
                "h_v": max(10, int(base_cap["mem"] * variance)),
                "d_v": round(random.uniform(0.5, 2.0), 1),
                "cost_c": round(random.uniform(0.5, 2.0), 2),
                "cost_r": round(random.uniform(0.5, 1.5), 2),
                "cost_h": round(random.uniform(0.5, 1.0), 2),
            }
        else:
            V[name] = {"server": False}
    return V


def generate_links(all_nodes, links_raw, difficulty: str) -> list:
    """
    Tạo links với BW đủ để đảm bảo routing feasible.
    difficulty ảnh hưởng BW variance.
    """
    bw_choices = DIFFICULTY_CFG[difficulty]["bw"]
    # Min BW = max bw_choice * 3 để đảm bảo luôn có path
    min_bw = max(bw_choices) * 5.0

    result = []
    for u, v in links_raw:
        result.append({
            "u": f"v{u}",
            "v": f"v{v}",
            "b_l": random.choice([min_bw, min_bw * 2, min_bw * 4]),
            "d_l": round(random.uniform(0.5, 2.0), 1),
        })
    return result


# ---------------------------------------------------------------------------
# VNF type generation
# ---------------------------------------------------------------------------

def generate_vnf_types(num_types: int, difficulty: str) -> list:
    """
    VNF resource demand tỉ lệ với difficulty.
    Đảm bảo tổng demand của 1 request << capacity của 1 server node.
    """
    # easy: VNF nhỏ → nhiều VNF fit trong 1 node
    # hard: VNF lớn hơn → cần nhiều nodes
    resource_range = {
        "easy":   dict(cpu=(2, 5),  ram=(1, 4),  mem=(3, 8)),
        "normal": dict(cpu=(4, 10), ram=(3, 8),  mem=(6, 15)),
        "hard":   dict(cpu=(8, 20), ram=(6, 16), mem=(12, 30)),
    }[difficulty]

    F = []
    for _ in range(num_types):
        F.append({
            "c_f": random.randint(*resource_range["cpu"]),
            "r_f": random.randint(*resource_range["ram"]),
            "h_f": random.randint(*resource_range["mem"]),
            "d_f": {},
        })
    return F


# ---------------------------------------------------------------------------
# Feasibility-aware request generation
# ---------------------------------------------------------------------------

def _compute_path_delay(G: nx.Graph, src: str, dst: str,
                         link_delays: dict) -> float:
    """Tính tổng delay của shortest path (theo hop)."""
    try:
        path = nx.shortest_path(G, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float('inf')
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += link_delays.get((u, v), link_delays.get((v, u), 1.0))
    return total


def _has_bw_path(G_bw: nx.Graph, src: str, dst: str) -> bool:
    """Kiểm tra tồn tại path với BW đủ."""
    try:
        nx.shortest_path(G_bw, src, dst)
        return True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False


def generate_requests(
    num_requests: int,
    server_nodes: list,
    all_links_raw: list,
    V: dict,
    F: list,
    difficulty: str,
    max_attempts_per_req: int = 20,
) -> list:
    """
    Tạo requests đảm bảo feasibility tối thiểu:
    1. src và dst connected qua path có BW đủ
    2. tổng VNF resource của request <= min server capacity
    3. delay_max (latency) >= actual path delay * lat_slack
    4. lifetime đủ dài để xử lý

    QUAN TRỌNG:
    - delay_max trong Request được dùng làm LIFETIME (end_time = arrival + delay_max)
    - latency_max là constraint riêng, lưu trong field "lat_max"
    - Tránh dual-use bug của generate.py gốc
    """
    cfg     = DIFFICULTY_CFG[difficulty]
    bw_list = cfg["bw"]
    lt_mul  = cfg["lt_mul"]
    lat_slk = cfg["lat_slack"]

    # Build networkx graph để kiểm tra connectivity
    G_full = nx.Graph()

    link_delays = {}
    for link_obj in all_links_raw:
        u, v = link_obj["u"], link_obj["v"]
        G_full.add_edge(u, v)
        link_delays[(u, v)] = link_obj["d_l"]
        link_delays[(v, u)] = link_obj["d_l"]

    # Tính server capacity tối thiểu để bound VNF demand
    server_caps = {}
    for name, vdata in V.items():
        if vdata.get("server"):
            server_caps[name] = {
                "cpu": vdata["c_v"],
                "ram": vdata["r_v"],
                "mem": vdata["h_v"],
            }

    if not server_caps or len(server_nodes) < 2:
        return []

    min_cap = {
        "cpu": min(c["cpu"] for c in server_caps.values()),
        "ram": min(c["ram"] for c in server_caps.values()),
        "mem": min(c["mem"] for c in server_caps.values()),
    }

    server_names = [f"v{n}" for n in server_nodes
                    if f"v{n}" in server_caps]
    if len(server_names) < 2:
        return []

    R          = []
    arrival_t  = 0.0
    failed     = 0

    for i in range(num_requests):
        placed = False
        for attempt in range(max_attempts_per_req):
            src = random.choice(server_names)
            dst = random.choice([s for s in server_names if s != src])
            bw  = random.choice(bw_list)

            # --- Check 1: BW-feasible path tồn tại ---
            G_bw = nx.Graph()
            for link_obj in all_links_raw:
                if link_obj["b_l"] >= bw:
                    G_bw.add_edge(link_obj["u"], link_obj["v"])
            if not _has_bw_path(G_bw, src, dst):
                continue

            # --- Check 2: path delay ---
            path_delay = _compute_path_delay(G_full, src, dst, link_delays)
            if path_delay == float('inf'):
                continue

            # --- Check 3: chọn VNFs với tổng resource hợp lý ---
            num_vnfs = random.randint(2, 4)
            vnf_indices = random.choices(range(len(F)), k=num_vnfs)
            total_res = {
                "cpu": sum(F[j]["c_f"] for j in vnf_indices),
                "ram": sum(F[j]["r_f"] for j in vnf_indices),
                "mem": sum(F[j]["h_f"] for j in vnf_indices),
            }

            # Tổng demand không vượt quá 70% capacity tối thiểu của 1 server
            if any(total_res[k] > min_cap[k] * 0.7 for k in ["cpu", "ram", "mem"]):
                continue

            # --- Latency constraint: actual path delay * slack ---
            # lat_max là latency budget (path delay phải <= lat_max)
            # TÁCH BIỆT với delay_max (lifetime)
            lat_max     = round(path_delay * lat_slk + num_vnfs * 1.0, 1)

            # Lifetime: đủ dài để xử lý (không liên quan đến path delay)
            lifetime    = round(num_vnfs * lt_mul + random.uniform(0, 5), 1)

            # Arrival time: staggered với jitter
            arrival_t  += random.uniform(3.0, 8.0)

            R.append({
                "T":       round(arrival_t, 2),
                "st_r":    src,
                "d_r":     dst,
                "F_r":     vnf_indices,
                "b_r":     bw,
                "d_max":   lifetime,    # lifetime — dùng để tính end_time
                "lat_max": lat_max,     # latency budget — dùng cho path delay check
            })
            placed = True
            break

        if not placed:
            failed += 1

    if failed > 0:
        print(f"  [WARN] {failed}/{num_requests} requests could not be made feasible", flush=True)
    return R


# ---------------------------------------------------------------------------
# Feasibility report
# ---------------------------------------------------------------------------

def compute_feasibility_report(data: dict) -> dict:
    """
    Ước tính tỉ lệ feasible requests mà không cần chạy full simulation.
    """
    V      = data["V"]
    E      = data["E"]
    F      = data["F"]
    R      = data["R"]

    # Build graph
    G   = nx.Graph()
    bws = {}
    for link in E:
        u, v = link["u"], link["v"]
        G.add_edge(u, v)
        bws[(u, v)] = link["b_l"]
        bws[(v, u)] = link["b_l"]

    server_caps = {
        name: {"cpu": d["c_v"], "ram": d["r_v"], "mem": d["h_v"]}
        for name, d in V.items() if d.get("server")
    }

    feasible = routing_ok = resource_ok = 0

    for req in R:
        src  = req["st_r"]
        dst  = req["d_r"]
        bw   = req["b_r"]
        vnfs = req["F_r"]

        # Routing check
        G_bw = nx.Graph()
        for link in E:
            if link["b_l"] >= bw:
                G_bw.add_edge(link["u"], link["v"])
        try:
            nx.shortest_path(G_bw, src, dst)
            r_ok = True
        except Exception:
            r_ok = False

        # Resource check: ít nhất 1 server có đủ resource cho mỗi VNF
        res_ok = all(
            any(
                server_caps[s]["cpu"] >= F[j]["c_f"] and
                server_caps[s]["ram"] >= F[j]["r_f"] and
                server_caps[s]["mem"] >= F[j]["h_f"]
                for s in server_caps
            )
            for j in vnfs
        )

        if r_ok:
            routing_ok += 1
        if res_ok:
            resource_ok += 1
        if r_ok and res_ok:
            feasible += 1

    n = len(R)
    return {
        "total":        n,
        "routing_ok":   routing_ok,
        "resource_ok":  resource_ok,
        "feasible_est": feasible,
        "feasible_pct": round(feasible / max(n, 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_single_file(topology: str, distribution: str, difficulty: str,
                          scale: int, num_requests: int,
                          seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    topo      = TOPOLOGIES[topology]
    all_nodes = topo["nodes"]
    raw_links = topo.get("links") or generate_grid_links(
        len(all_nodes), extra_link_ratio=0.6)

    G = build_nx_graph(raw_links)

    # Chọn server nodes dựa trên degree thực tế
    server_nodes     = select_server_nodes(G, distribution)
    server_nodes_set = set(server_nodes)

    V = generate_nodes(all_nodes, server_nodes_set, scale, difficulty)
    E = generate_links(all_nodes, raw_links, difficulty)

    max_vnf = 5
    F       = generate_vnf_types(max_vnf, difficulty)
    R       = generate_requests(num_requests, server_nodes, E, V, F, difficulty)

    return {"V": V, "E": E, "F": F, "R": R}


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility-guaranteed NFV data generator")
    parser.add_argument("--topology",     default="nsf",
                        choices=["nsf", "conus", "cogent"])
    parser.add_argument("--distribution", default="rural",
                        choices=["uniform", "rural", "urban", "centers"])
    parser.add_argument("--difficulty",   default="easy",
                        choices=["easy", "normal", "hard"])
    parser.add_argument("--scale",        type=int, default=50)
    parser.add_argument("--num-files",    type=int, default=1)
    parser.add_argument("--requests",     type=int, default=50)
    parser.add_argument("--output",       default="data")
    parser.add_argument("--seed-offset",  type=int, default=0)
    parser.add_argument("--report",       action="store_true",
                        help="In feasibility report sau khi generate")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for i in range(args.num_files):
        seed = 42 + args.seed_offset + i
        data = generate_single_file(
            topology=args.topology,
            distribution=args.distribution,
            difficulty=args.difficulty,
            scale=args.scale,
            num_requests=args.requests,
            seed=seed,
        )

        filename = (f"{args.topology}_{args.distribution}_"
                    f"{args.difficulty}_s{i+1}.json")
        filepath = os.path.join(args.output, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        server_count = sum(1 for v in data["V"].values() if v.get("server"))
        print(f"Generated: {filepath}")
        print(f"  Nodes: {len(data['V'])}  Links: {len(data['E'])}"
              f"  Requests: {len(data['R'])}  Servers: {server_count}")

        if args.report:
            rpt = compute_feasibility_report(data)
            print(f"  Feasibility estimate: {rpt['feasible_est']}/{rpt['total']}"
                  f" ({rpt['feasible_pct']}%)"
                  f"  [routing={rpt['routing_ok']} resource={rpt['resource_ok']}]")


if __name__ == "__main__":
    main()