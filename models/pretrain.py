from __future__ import annotations

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config
from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from models.model import VGAENetwork, LowLevelAgent, ReplayBuffer

LATENT_DIM = 8
MAX_DCS    = 60
VGAE_DIR   = "models/vgae_pretrained"
LL_DIR     = "models/ll_pretrained"


def load_env(path: str) -> Env:
    with open(path) as f:
        data = json.load(f)
    network  = Network()
    vnfs     = ListOfVnfs()
    requests = ListOfRequests()

    for nid, nd in data.get("V", {}).items():
        if nd.get("server", False):
            network.add_dc_node(
                name=nid, delay=nd.get("d_v", 0.0),
                capacity={"mem": nd.get("h_v", 1.), "cpu": nd.get("c_v", 1.), "ram": nd.get("r_v", 1.)},
                cost={"mem": nd.get("cost_h", 1.), "cpu": nd.get("cost_c", 1.), "ram": nd.get("cost_r", 1.)})
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]),
                         lnk.get("b_l", 1.0), lnk.get("d_l", 1.0))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f", 1.), c_f=vd.get("c_f", 1.), r_f=vd.get("r_f", 1.),
                         d_f={k: v for k, v in vd.get("d_f", {}).items()}))

    for idx, rd in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T", 0),
            delay_max=rd.get("d_max", 100.),
            start_node=str(rd.get("st_r", "")), end_node=str(rd.get("d_r", "")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r", [])],
            bandwidth=rd.get("b_r", 1.)))
    return Env(network, vnfs, requests)


def build_dc_graph(env: Env, t_start: int, t_end: int, bw: float,
                   path_cache: dict = None):
    """
    Build DC feature matrix X, adjacency A, and dc list.
    path_cache: optional dict keyed by (t_start, t_end, round(bw,1)) → all_paths
    """
    import networkx as nx

    dcs = [nid for nid, n in env.network.nodes.items() if n.type == config.NODE_DC]
    n   = len(dcs)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 0), np.float32), []

    # --- pruned graph (bandwidth-feasible edges) ---
    G = nx.Graph()
    for nid in env.network.nodes:
        G.add_node(nid)
    for lnk in env.network.links:
        if lnk.get_available_bandwidth(t_start, t_end) >= bw:
            G.add_edge(lnk.u.name, lnk.v.name, delay=lnk.delay)

    # --- path lengths (cached) ---
    cache_key = (t_start, t_end, round(bw, 1))
    if path_cache is not None and cache_key in path_cache:
        all_len = path_cache[cache_key]
    else:
        try:
            all_len = dict(nx.shortest_path_length(G, weight="delay"))
        except Exception:
            all_len = {}
        if path_cache is not None:
            path_cache[cache_key] = all_len

    # --- normalisation ---
    max_r = {k: 1.0 for k in config.RESOURCE_TYPE}
    for nd in env.network.nodes.values():
        if nd.type == config.NODE_DC and nd.cap:
            for k in config.RESOURCE_TYPE:
                max_r[k] = max(max_r[k], nd.cap[k])

    X = np.zeros((n, 3), np.float32)
    A = np.zeros((n, n), np.float32)
    for i, did in enumerate(dcs):
        res  = env.network.nodes[did].get_min_available_resource(t_start, t_end)
        X[i] = [res[k] / max_r[k] for k in config.RESOURCE_TYPE]
        for j, dj in enumerate(dcs):
            if i == j:
                A[i, j] = 1.0
            else:
                d = all_len.get(did, {}).get(dj, None)
                if d is not None and d > 0:
                    A[i, j] = 1.0 / (d + 1.0)
    return X, A, dcs


# ---------------------------------------------------------------------------
# PHASE 1 — VGAE pre-training
# ---------------------------------------------------------------------------

def pretrain_vgae(train_dir: str, epochs: int = 200, batch: int = 16):
    files = sorted([os.path.join(train_dir, f)
                    for f in os.listdir(train_dir) if f.endswith(".json")])
    if not files:
        print(f"[Pretrain-VGAE] No JSON files in {train_dir}")
        return None

    print(f"\n{'='*50}")
    print(f"PHASE 1: VGAE Pre-training  ({len(files)} files, {epochs} epochs)")
    print(f"{'='*50}")

    vgae       = VGAENetwork(latent_dim=LATENT_DIM)
    buf        = ReplayBuffer(capacity=2000)
    path_cache = {}   # shared across all files in phase 1

    print("Collecting graph snapshots ...")
    for fp in files:
        env = load_env(fp)
        env.reset()
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) >= 2:
                buf.push((X, A))
    print(f"  Collected {len(buf)} graph snapshots")

    if len(buf) < 4:
        print("[Pretrain-VGAE] Too few snapshots, skipping")
        return None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        vgae.train(buf, epochs=1, batch=batch)
        if ep % 20 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}  ({time.time()-t0:.1f}s)")

    os.makedirs(VGAE_DIR, exist_ok=True)
    out = os.path.join(VGAE_DIR, "vgae_weights.npy")
    vgae.save_weights(out)
    print(f"[Pretrain-VGAE] Saved → {out}")
    return vgae


# ---------------------------------------------------------------------------
# PHASE 2 — LL-DQN pre-training (imitation from BestFit)
# ---------------------------------------------------------------------------

def pretrain_ll(train_dir: str, vgae: VGAENetwork,
                episodes: int = 200, batch: int = 32):
    files = sorted([os.path.join(train_dir, f)
                    for f in os.listdir(train_dir) if f.endswith(".json")])
    if not files:
        print(f"[Pretrain-LL] No JSON files in {train_dir}")
        return None

    print(f"\n{'='*50}")
    print(f"PHASE 2: LL-DQN Pre-training (imitation)  ({episodes} ep)")
    print(f"{'='*50}")

    ll_agent = LowLevelAgent(latent_dim=LATENT_DIM, max_dcs=MAX_DCS,
                              input_dim=LATENT_DIM + 3)
    buf_LL = ReplayBuffer(capacity=20_000)

    # Import teacher once
    from strategy.best_fit import BestFit
    from env.request import SFC as SFCcls

    t0           = time.time()
    prev_fp      = None
    path_cache   = {}   # per-file path cache, cleared on file change

    for ep in range(1, episodes + 1):
        fp  = files[(ep - 1) % len(files)]
        env = load_env(fp)
        env.reset()

        # Clear path cache when switching to a different file
        if fp != prev_fp:
            path_cache.clear()
            prev_fp = fp

        teacher = BestFit(env)

        # ----------------------------------------------------------------
        # Collect all transitions for this episode first (data pass),
        # then train — avoids interleaving Python/TF calls per request.
        # ----------------------------------------------------------------
        transitions = []

        for req in sorted(env.requests, key=lambda r: r.arrival_time):
            sfc = SFCcls(req)
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)

            # Build DC graph with shared path cache (no redundant Dijkstra)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) == 0:
                continue

            Z = vgae.encode(X, A)

            plan = teacher.get_placement(sfc, req.arrival_time)
            if plan is None:
                continue

            for k, vnf in enumerate(req.vnfs):
                node_plan = plan.get("nodes", {}).get(str(k))
                if node_plan is None:
                    continue
                greedy_dc = node_plan["dc"]
                if greedy_dc not in dcs:
                    continue
                act_idx = dcs.index(greedy_dc)
                if act_idx >= MAX_DCS:
                    continue

                vnf_feat = np.array(
                    [vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE],
                    dtype=np.float32)

                valid = [i for i, d in enumerate(dcs)
                         if i < MAX_DCS and env._check_can_deploy_vnf(
                             env.network.nodes[d], vnf, t_s, t_e)]
                if not valid:
                    continue

                transitions.append((Z, vnf_feat, act_idx, valid))

            # Apply plan to update resource state (so next requests see
            # correct remaining capacity — same as the real env does)
            env.step(plan)

        # Push all transitions into replay buffer
        for Z, vnf_feat, act_idx, valid in transitions:
            buf_LL.push((Z, vnf_feat, act_idx, 1.0, Z, valid, False))

        # Train: one sweep over buffer, capped to avoid over-fitting per ep
        if len(buf_LL) >= batch:
            n_batches = min(len(buf_LL) // batch, 15)
            for _ in range(n_batches):
                ll_agent.train(buf_LL, batch)

        if ep % 20 == 0 or ep == episodes:
            elapsed = time.time() - t0
            eta     = (episodes - ep) * (elapsed / ep)
            eta_str = (f"{eta/3600:.1f}h" if eta > 3600
                       else f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s")
            print(f"  ep {ep:>3}/{episodes}  buffer={len(buf_LL)}  "
                  f"elapsed={elapsed:.1f}s  ETA={eta_str}")

        env.reset()

    os.makedirs(LL_DIR, exist_ok=True)
    out = os.path.join(LL_DIR, "ll_dqn_weights.weights.h5")
    ll_agent.policy_net.save_weights(out)
    print(f"[Pretrain-LL] Saved → {out}")
    return ll_agent