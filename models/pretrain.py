"""
models/pretrain.py  –  Pre-training pipeline for HRL-VGAE

Phase 1 – VGAE pre-training
  • Load training JSON files, build DC-full graphs.
  • Train VGAE to reconstruct adjacency matrices (unsupervised).
  • Save to models/vgae_pretrained/vgae_weights.npy

Phase 2 – Low-Level DQN pre-training
  • Use the frozen pre-trained VGAE encoder.
  • Run Greedy-FIFS as a teacher: collect (state, greedy_action) pairs.
  • Supervised imitation learning → warm-start LL policy.
  • Save to models/ll_pretrained/ll_dqn_weights.weights.h5

Both phases are fast: < 30 min total on CPU.
"""

from __future__ import annotations

import os, sys, json, argparse, copy, time
import numpy as np

# ── path fix ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

import config
from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from models.model import VGAENetwork, LowLevelAgent, ReplayBuffer
from strategy.fifs import GreedyFIFS

LATENT_DIM   = 8
MAX_DCS      = 60
VGAE_DIR     = "models/vgae_pretrained"
LL_DIR       = "models/ll_pretrained"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

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
                capacity={"mem": nd.get("h_v",1.), "cpu": nd.get("c_v",1.), "ram": nd.get("r_v",1.)},
                cost={"mem": nd.get("cost_h",1.), "cpu": nd.get("cost_c",1.), "ram": nd.get("cost_r",1.)})
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]),
                         lnk.get("b_l", 1.0), lnk.get("d_l", 1.0))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f",1.), c_f=vd.get("c_f",1.), r_f=vd.get("r_f",1.),
                         d_f={k:v for k,v in vd.get("d_f",{}).items()}))

    for idx, rd in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T",0),
            delay_max=rd.get("d_max",100.),
            start_node=str(rd.get("st_r","")), end_node=str(rd.get("d_r","")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r",[])],
            bandwidth=rd.get("b_r",1.)))
    return Env(network, vnfs, requests)


def build_dc_graph(env: Env, t_start: int, t_end: int, bw: float):
    import networkx as nx
    dcs = [nid for nid, n in env.network.nodes.items() if n.type == config.NODE_DC]
    n   = len(dcs)
    if n == 0:
        return np.zeros((0,3),np.float32), np.zeros((0,0),np.float32), []

    G = nx.Graph()
    for nid in env.network.nodes:
        G.add_node(nid)
    for lnk in env.network.links:
        if lnk.get_available_bandwidth(t_start, t_end) >= bw:
            G.add_edge(lnk.u.name, lnk.v.name, delay=lnk.delay)

    try:
        all_len = dict(nx.shortest_path_length(G, weight="delay"))
    except Exception:
        all_len = {}

    max_r = {k:1.0 for k in config.RESOURCE_TYPE}
    for nd in env.network.nodes.values():
        if nd.type == config.NODE_DC and nd.cap:
            for k in config.RESOURCE_TYPE:
                max_r[k] = max(max_r[k], nd.cap[k])

    X = np.zeros((n,3), np.float32)
    A = np.zeros((n,n), np.float32)
    for i, did in enumerate(dcs):
        res = env.network.nodes[did].get_min_available_resource(t_start, t_end)
        X[i] = [res[k]/max_r[k] for k in config.RESOURCE_TYPE]
        for j, dj in enumerate(dcs):
            if i == j:
                A[i,j] = 1.0
            else:
                d = all_len.get(did, {}).get(dj, None)
                if d is not None and d > 0:
                    A[i,j] = 1.0 / (d+1.0)
    return X, A, dcs


# ─────────────────────────────────────────────────────────────
# Phase 1: VGAE pre-training
# ─────────────────────────────────────────────────────────────

def pretrain_vgae(train_dir: str, epochs: int = 200, batch: int = 16):
    files = sorted([os.path.join(train_dir, f)
                    for f in os.listdir(train_dir) if f.endswith(".json")])
    if not files:
        print(f"[Pretrain-VGAE] No JSON files in {train_dir}")
        return

    print(f"\n{'='*50}")
    print(f"PHASE 1: VGAE Pre-training  ({len(files)} files, {epochs} epochs)")
    print(f"{'='*50}")

    vgae   = VGAENetwork(latent_dim=LATENT_DIM)
    buf    = ReplayBuffer(capacity=2000)

    # Collect graph snapshots
    print("Collecting graph snapshots …")
    for fp in files:
        env = load_env(fp)
        env.reset()
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw)
            if len(dcs) >= 2:
                buf.push((X, A))
    print(f"  Collected {len(buf)} graph snapshots")

    if len(buf) < 4:
        print("[Pretrain-VGAE] Too few snapshots, skipping")
        return

    t0 = time.time()
    for ep in range(1, epochs+1):
        vgae.train(buf, epochs=1, batch=batch)
        if ep % 50 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}  ({time.time()-t0:.1f}s)")

    os.makedirs(VGAE_DIR, exist_ok=True)
    out = os.path.join(VGAE_DIR, "vgae_weights.npy")
    vgae.save_weights(out)
    print(f"[Pretrain-VGAE] Saved → {out}")
    return vgae


# ─────────────────────────────────────────────────────────────
# Phase 2: Low-Level DQN pre-training (imitation from FIFS)
# ─────────────────────────────────────────────────────────────

def pretrain_ll(train_dir: str, vgae: VGAENetwork,
                episodes: int = 200, batch: int = 32):
    files = sorted([os.path.join(train_dir, f)
                    for f in os.listdir(train_dir) if f.endswith(".json")])
    if not files:
        print(f"[Pretrain-LL] No JSON files in {train_dir}")
        return

    print(f"\n{'='*50}")
    print(f"PHASE 2: LL-DQN Pre-training (imitation)  ({episodes} ep)")
    print(f"{'='*50}")

    ll_agent = LowLevelAgent(latent_dim=LATENT_DIM, max_dcs=MAX_DCS,
                              input_dim=LATENT_DIM + 3)
    buf_LL = ReplayBuffer(capacity=20_000)

    t0 = time.time()
    for ep in range(1, episodes+1):
        fp  = files[(ep-1) % len(files)]
        env = load_env(fp)
        env.reset()

        fifs = GreedyFIFS(env)
        env.set_strategy(fifs)

        from env.request import SFC as SFCcls
        for req in sorted(env.requests, key=lambda r: r.arrival_time):
            sfc = SFCcls(req)
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)

            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw)
            if len(dcs) == 0:
                continue
            Z = vgae.encode(X, A)

            # Get FIFS placement as teacher signal
            plan = fifs.get_placement(sfc, req.arrival_time)
            if plan is None:
                continue

            # For each VNF, record (state, greedy_dc_idx) as an imitation sample
            for k, vnf in enumerate(req.vnfs):
                if str(k) not in plan.get("nodes", {}):
                    continue
                greedy_dc = plan["nodes"][str(k)]["dc"]
                if greedy_dc not in dcs:
                    continue
                act_idx = dcs.index(greedy_dc)
                if act_idx >= MAX_DCS:
                    continue
                vnf_feat = [vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE]
                valid = [i for i, d in enumerate(dcs)
                         if i < MAX_DCS and env._check_can_deploy_vnf(
                             env.network.nodes[d], vnf, t_s, t_e)]
                if not valid:
                    continue
                # Imitation: reward = +1 for greedy action, 0 otherwise
                buf_LL.push((Z, np.array([vnf_feat],np.float32), act_idx,
                             1.0, Z, valid, False))

            # Apply plan so next VNF sees updated resources
            env.step(plan)

        # Train from buffer
        if len(buf_LL) >= batch:
            for _ in range(min(10, len(buf_LL) // batch)):
                ll_agent.train(buf_LL, batch)

        if ep % 50 == 0 or ep == episodes:
            print(f"  ep {ep}/{episodes}  buffer={len(buf_LL)}  "
                  f"({time.time()-t0:.1f}s)")

    os.makedirs(LL_DIR, exist_ok=True)
    out = os.path.join(LL_DIR, "ll_dqn_weights.weights.h5")
    ll_agent.policy_net.save_weights(out)
    print(f"[Pretrain-LL] Saved → {out}")
    return ll_agent


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",       default="both",
                    choices=["vgae","ll","both"])
    ap.add_argument("--train-dir",   default="data/train")
    ap.add_argument("--vgae-epochs", type=int, default=200)
    ap.add_argument("--ll-episodes", type=int, default=200)
    args = ap.parse_args()

    vgae = None
    if args.phase in ("vgae", "both"):
        vgae = pretrain_vgae(args.train_dir, epochs=args.vgae_epochs)

    if args.phase in ("ll", "both"):
        if vgae is None:
            vgae = VGAENetwork(latent_dim=LATENT_DIM)
            wp   = os.path.join(VGAE_DIR, "vgae_weights.npy")
            if os.path.exists(wp):
                vgae.load_weights(wp)
                print(f"[Pretrain-LL] Loaded VGAE from {wp}")
        pretrain_ll(args.train_dir, vgae, episodes=args.ll_episodes)

    print("\nPre-training complete.")


if __name__ == "__main__":
    main()