"""
Synthetic data generator for NFV VNF Placement.

Usage:
    python data/generate.py --topology nsf --distribution rural --difficulty easy --num-files 10 --scale 50
    python data/generate.py --topology cogent --distribution centers --difficulty hard --num-files 5 --scale 100
"""

import os
import sys
import json
import random
import argparse
import numpy as np

TOPOLOGIES = {
    "nsf": {
        "nodes": list(range(14)),
        "links": [
            (0,1),(0,3),(0,4),(1,2),(1,7),(2,3),(2,10),(3,12),
            (4,5),(4,8),(5,6),(5,9),(5,12),(6,9),(6,11),(7,11),
            (7,13),(8,10),(8,13),(9,13),(10,12),(11,13)
        ],
        "node_names": [f"v{i}" for i in range(14)]
    },
    "conus": {
        "nodes": list(range(75)),
        "node_names": [f"v{i}" for i in range(75)]
    },
    "cogent": {
        "nodes": list(range(197)),
        "node_names": [f"v{i}" for i in range(197)]
    }
}


def generate_grid_links(n_nodes, extra_link_ratio=0.5):
    """Generate random graph with connectivity like a grid + extra links."""
    n = n_nodes
    links = set()
    # Ensure connected: create a spanning chain
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(n - 1):
        links.add((min(perm[i], perm[i+1]), max(perm[i], perm[i+1])))
    # Add extra random links
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in links]
    n_extra = int(n * extra_link_ratio)
    for i, j in random.sample(all_pairs, min(n_extra, len(all_pairs))):
        links.add((min(i, j), max(i, j)))
    return [(i, j) for i, j in sorted(links)]


def select_server_nodes(topo_name, nodes, distribution, fraction=0.3):
    """Select server nodes based on distribution strategy."""
    n = len(nodes)
    if distribution == "uniform":
        return random.sample(nodes, max(1, int(n * fraction)))
    elif distribution == "rural":
        # 70% lowest degree nodes
        return random.sample(nodes, max(1, int(n * 0.7)))
    elif distribution == "urban":
        # 30% highest degree nodes
        return nodes[:max(1, int(n * 0.3))]
    elif distribution == "centers":
        # 10% highest degree nodes
        return nodes[:max(1, int(n * 0.1))]
    return random.sample(nodes, max(1, int(n * fraction)))


def generate_nodes(all_nodes, server_nodes, scale):
    """Generate V and E sections with resource values."""
    V = {}
    base_resources = {"mem": 100, "cpu": 100, "ram": 64}
    node_names = TOPOLOGIES["nsf"]["node_names"]
    if len(all_nodes) > 14:
        node_names = [f"v{i}" for i in range(len(all_nodes))]

    for node_id in all_nodes:
        name = node_names[node_id] if node_id < len(node_names) else f"v{node_id}"
        if node_id in server_nodes:
            scale_factor = scale / 50.0
            V[name] = {
                "server": True,
                "c_v": int(base_resources["cpu"] * scale_factor),
                "r_v": int(base_resources["ram"] * scale_factor),
                "h_v": int(base_resources["mem"] * scale_factor),
                "d_v": round(random.uniform(0.5, 2.0), 1),
                "cost_c": round(random.uniform(0.5, 2.0), 2),
                "cost_r": round(random.uniform(0.5, 1.5), 2),
                "cost_h": round(random.uniform(0.5, 1.0), 2),
            }
        else:
            V[name] = {"server": False}

    return V


def generate_links(all_nodes, topo_name, links=None):
    """Generate E section."""
    if links is None:
        links = generate_grid_links(len(all_nodes), extra_link_ratio=0.5)

    result = []
    for u, v in links:
        if topo_name == "nsf" and u < 14 and v < 14:
            d_v = 1.0
        else:
            d_v = round(random.uniform(1.0, 5.0), 1)

        result.append({
            "u": f"v{u}",
            "v": f"v{v}",
            "b_l": random.choice([10.0, 20.0, 40.0, 100.0]),
            "d_l": d_v
        })
    return result


def generate_requests(num_requests, server_nodes, links, difficulty, vnfs_per_req_range=(2, 5)):
    """Generate R section."""
    R = []
    node_names = [f"v{i}" for i in server_nodes]

    if not node_names or len(node_names) < 2:
        return R

    for i in range(num_requests):
        st = random.choice(node_names)
        dst = random.choice([n for n in node_names if n != st])
        num_vnfs = random.randint(*vnfs_per_req_range)

        base_bw = 1.0 if difficulty == "easy" else random.choice([1.0, 2.0, 5.0])
        delay_max = max(num_vnfs * 3, 10) if difficulty == "easy" else max(num_vnfs * 2, 5)

        R.append({
            "T": i * 5 + random.uniform(0, 3),
            "st_r": st,
            "d_r": dst,
            "F_r": list(range(num_vnfs)),
            "b_r": base_bw,
            "d_max": delay_max
        })

    return R


def generate_vnf_types(num_vnfs):
    """Generate F section - VNF type definitions."""
    F = []
    for i in range(num_vnfs):
        F.append({
            "c_f": random.choice([5, 10, 15]),
            "r_f": random.choice([4, 8, 12]),
            "h_f": random.choice([10, 20, 30]),
            "d_f": {}
        })
    return F


def generate_single_file(topology, distribution, difficulty, scale, num_requests, seed=None):
    """Generate a single data file."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    topo = TOPOLOGIES[topology]
    all_nodes = topo["nodes"]
    base_links = topo.get("links", None)

    # For large topologies, generate links
    if base_links is None:
        base_links = generate_grid_links(len(all_nodes), extra_link_ratio=0.5)

    # Select server nodes
    server_nodes = select_server_nodes(topology, all_nodes, distribution)

    # Generate components
    V = generate_nodes(all_nodes, server_nodes, scale)
    E = generate_links(all_nodes, topology, base_links)
    max_vnf = 5
    F = generate_vnf_types(max_vnf)
    R = generate_requests(num_requests, server_nodes, base_links, difficulty)

    return {"V": V, "E": E, "F": F, "R": R}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NFV data")
    parser.add_argument("--topology", type=str, default="nsf",
                        choices=["nsf", "conus", "cogent"],
                        help="Network topology")
    parser.add_argument("--distribution", type=str, default="rural",
                        choices=["uniform", "rural", "urban", "centers"],
                        help="Server node distribution")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "hard"],
                        help="Difficulty level")
    parser.add_argument("--scale", type=int, default=50,
                        help="Resource scale factor (bigger = more resources)")
    parser.add_argument("--num-files", type=int, default=1,
                        help="Number of files to generate")
    parser.add_argument("--requests", type=int, default=50,
                        help="Number of requests per file")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset to avoid file overlap")
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
            seed=seed
        )

        filename = f"{args.topology}_{args.distribution}_{args.difficulty}_s{i+1}.json"
        filepath = os.path.join(args.output, filename)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Generated: {filepath}")
        print(f"  Nodes: {len(data['V'])}, Links: {len(data['E'])}, Requests: {len(data['R'])}")
        server_count = sum(1 for v in data['V'].values() if v.get('server', False))
        print(f"  Server nodes: {server_count}")


if __name__ == "__main__":
    main()