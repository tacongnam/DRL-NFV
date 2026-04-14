from __future__ import annotations

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from models.model import VGAENetwork, LowLevelAgent, ReplayBuffer

LATENT_DIM = 8
MAX_DCS = 60
VGAE_DIR = "models/vgae_pretrained"
LL_DIR = "models/ll_pretrained"
VGAE_WEIGHTS_FILE = "vgae_weights.npy"
LL_WEIGHTS_FILE = "ll_dqn_weights.npy"
DEFAULT_PROFILE = "fast"
DEFAULT_MAX_FILES = 3
DEFAULT_MAX_REQUESTS = 160
DEFAULT_VGAE_EPOCHS = 60
DEFAULT_LL_EPISODES = 60
DEFAULT_MIN_SERVERS = 3
DEFAULT_MIN_SERVER_RATIO = 0.12
DEFAULT_SAMPLE_MODE = "uniform"


def sample_requests(req_rows: list, max_requests: int = None,
                    sample_mode: str = DEFAULT_SAMPLE_MODE) -> list:
    if max_requests is None or max_requests <= 0 or len(req_rows) <= max_requests:
        return req_rows
    if sample_mode == "head":
        return req_rows[:max_requests]
    idxs = np.linspace(0, len(req_rows) - 1, num=max_requests, dtype=int)
    return [req_rows[i] for i in idxs]


def load_env(path: str, max_requests: int = None,
             sample_mode: str = DEFAULT_SAMPLE_MODE) -> Env:
    with open(path) as f:
        data = json.load(f)
    network = Network()
    vnfs = ListOfVnfs()
    requests = ListOfRequests()

    for nid, nd in data.get("V", {}).items():
        if nd.get("server", False):
            network.add_dc_node(
                name=nid,
                delay=nd.get("d_v", 0.0),
                capacity={"mem": nd.get("h_v", 1.0), "cpu": nd.get("c_v", 1.0), "ram": nd.get("r_v", 1.0)},
                cost={"mem": nd.get("cost_h", 1.0), "cpu": nd.get("cost_c", 1.0), "ram": nd.get("cost_r", 1.0)},
            )
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]), lnk.get("b_l", 1.0), lnk.get("d_l", 1.0))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(
            VNF(
                idx,
                h_f=vd.get("h_f", 1.0),
                c_f=vd.get("c_f", 1.0),
                r_f=vd.get("r_f", 1.0),
                d_f={k: v for k, v in vd.get("d_f", {}).items()},
            )
        )

    req_rows = sorted(data.get("R", []), key=lambda r: r.get("T", 0))
    req_rows = sample_requests(req_rows, max_requests=max_requests, sample_mode=sample_mode)

    for idx, rd in enumerate(req_rows):
        requests.add_request(
            Request(
                name=idx,
                arrival_time=rd.get("T", 0),
                delay_max=rd.get("d_max", 100.0),
                start_node=str(rd.get("st_r", "")),
                end_node=str(rd.get("d_r", "")),
                VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r", [])],
                bandwidth=rd.get("b_r", 1.0),
            )
        )
    return Env(network, vnfs, requests)


def read_dataset_meta(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    requests = data.get("R", [])
    total_requests = len(requests)
    server_count = sum(1 for node in data.get("V", {}).values() if node.get("server", False))
    node_count = len(data.get("V", {}))
    avg_bw = (sum(r.get("b_r", 0.0) for r in requests) / total_requests) if total_requests else 0.0
    avg_delay = (sum(r.get("d_max", 0.0) for r in requests) / total_requests) if total_requests else 0.0
    parts = os.path.basename(path).replace(".json", "").split("_")
    topology = parts[0] if parts else "unknown"
    difficulty = parts[2] if len(parts) > 2 else "unknown"
    return {
        "path": path,
        "name": os.path.basename(path),
        "topology": topology,
        "difficulty": difficulty,
        "nodes": node_count,
        "servers": server_count,
        "server_ratio": (server_count / node_count) if node_count else 0.0,
        "requests": total_requests,
        "avg_bw": avg_bw,
        "avg_delay": avg_delay,
    }


def select_train_files(train_dir: str, profile: str = DEFAULT_PROFILE,
                       max_files: int = DEFAULT_MAX_FILES,
                       min_servers: int = DEFAULT_MIN_SERVERS,
                       min_server_ratio: float = DEFAULT_MIN_SERVER_RATIO) -> list:
    files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".json")])
    metas = [read_dataset_meta(fp) for fp in files]
    if profile == "full":
        selected = metas
    else:
        selected = [
            m for m in metas
            if m["difficulty"] == "easy"
            and m["servers"] >= min_servers
            and m["server_ratio"] >= min_server_ratio
        ]
        if not selected:
            selected = metas
        selected = sorted(
            selected,
            key=lambda m: (m["nodes"], -m["server_ratio"], m["avg_bw"], -m["avg_delay"], m["name"]),
        )
    if max_files and max_files > 0:
        selected = selected[:max_files]
    return selected


def print_selected_files(selected: list, max_requests: int):
    print(f"[Pretrain] Selected {len(selected)} file(s)", flush=True)
    for meta in selected:
        req_label = meta["requests"] if not max_requests else min(meta["requests"], max_requests)
        print(
            f"  - {meta['name']}: nodes={meta['nodes']} servers={meta['servers']}"
            f" req={req_label}/{meta['requests']} bw={meta['avg_bw']:.2f}"
            f" delay={meta['avg_delay']:.2f}",
            flush=True,
        )


def _add_pretrain_args(parser: argparse.ArgumentParser):
    parser.add_argument("--phase", type=str, default="both", choices=["vgae", "ll", "both"])
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--profile", type=str, default=DEFAULT_PROFILE, choices=["fast", "balanced", "full"])
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--sample-mode", type=str, default=DEFAULT_SAMPLE_MODE, choices=["uniform", "head"])
    parser.add_argument("--min-servers", type=int, default=DEFAULT_MIN_SERVERS)
    parser.add_argument("--min-server-ratio", type=float, default=DEFAULT_MIN_SERVER_RATIO)
    parser.add_argument("--vgae-epochs", type=int, default=DEFAULT_VGAE_EPOCHS)
    parser.add_argument("--ll-episodes", type=int, default=DEFAULT_LL_EPISODES)


def build_dc_graph(env: Env, t_start: int, t_end: int, bw: float, path_cache: dict = None):
    import networkx as nx

    dcs = [nid for nid, n in env.network.nodes.items() if n.type == config.NODE_DC]
    n = len(dcs)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 0), np.float32), []

    cache_key = (t_start, t_end, round(bw, 1))
    if path_cache is not None and cache_key in path_cache:
        all_len = path_cache[cache_key]
    else:
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
        if path_cache is not None:
            path_cache[cache_key] = all_len

    max_r = {k: 1.0 for k in config.RESOURCE_TYPE}
    for nd in env.network.nodes.values():
        if nd.type == config.NODE_DC and nd.cap:
            for k in config.RESOURCE_TYPE:
                max_r[k] = max(max_r[k], nd.cap[k])

    X = np.zeros((n, 3), np.float32)
    A = np.zeros((n, n), np.float32)
    for i, did in enumerate(dcs):
        res = env.network.nodes[did].get_min_available_resource(t_start, t_end)
        X[i] = [res[k] / max_r[k] for k in config.RESOURCE_TYPE]
        for j, dj in enumerate(dcs):
            if i == j:
                A[i, j] = 1.0
            else:
                dist = all_len.get(did, {}).get(dj, None)
                if dist is not None and dist > 0:
                    A[i, j] = 1.0 / (dist + 1.0)
    return X, A, dcs


def pretrain_vgae(train_files: list, epochs: int = 200, batch: int = 16,
                  max_requests: int = DEFAULT_MAX_REQUESTS,
                  sample_mode: str = DEFAULT_SAMPLE_MODE):
    if not train_files:
        print("[Pretrain-VGAE] No JSON files selected", flush=True)
        return None

    print(f"\n{'=' * 50}", flush=True)
    print(f"PHASE 1: VGAE Pre-training  ({len(train_files)} files, {epochs} epochs)", flush=True)
    print(f"{'=' * 50}", flush=True)

    vgae = VGAENetwork(latent_dim=LATENT_DIM)
    buf = ReplayBuffer(capacity=2000)
    path_cache = {}
    per_file_snapshots = {}

    print("Collecting graph snapshots ...", flush=True)
    for meta in train_files:
        env = load_env(meta["path"], max_requests=max_requests, sample_mode=sample_mode)
        env.reset()
        local_count = 0
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) >= 2:
                buf.push((X, A))
                local_count += 1
        per_file_snapshots[meta["name"]] = local_count
    print(f"  Collected {len(buf)} graph snapshots", flush=True)
    for name, count in per_file_snapshots.items():
        print(f"    {name}: {count}", flush=True)

    if len(buf) < 4:
        print("[Pretrain-VGAE] Too few snapshots, skipping", flush=True)
        return None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        vgae.train(buf, epochs=1, batch=batch)
        if ep % 50 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}  ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(VGAE_DIR, exist_ok=True)
    out = os.path.join(VGAE_DIR, VGAE_WEIGHTS_FILE)
    vgae.save_weights(out)
    print(f"[Pretrain-VGAE] Saved -> {out}", flush=True)
    return vgae


def pretrain_ll(train_files: list, vgae: VGAENetwork, episodes: int = 200, batch: int = 32,
                max_requests: int = DEFAULT_MAX_REQUESTS,
                sample_mode: str = DEFAULT_SAMPLE_MODE):
    if not train_files:
        print("[Pretrain-LL] No JSON files selected", flush=True)
        return None

    print(f"\n{'=' * 50}", flush=True)
    print(f"PHASE 2: LL-DQN Pre-training (imitation)  ({episodes} ep)", flush=True)
    print(f"{'=' * 50}", flush=True)

    ll_agent = LowLevelAgent(latent_dim=LATENT_DIM, max_dcs=MAX_DCS, input_dim=LATENT_DIM + 3)
    dummy = np.zeros((1, LATENT_DIM + 3), dtype=np.float32)
    ll_agent.policy_net(dummy)
    ll_agent.target_net(dummy)

    buf_ll = ReplayBuffer(capacity=20_000)

    print("Pre-building graph caches ...", flush=True)
    file_envs = []
    for meta in train_files:
        env = load_env(meta["path"], max_requests=max_requests, sample_mode=sample_mode)
        env.reset()
        path_cache = {}
        sorted_reqs = sorted(env.requests, key=lambda r: r.arrival_time)
        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            build_dc_graph(env, t_s, t_e, req.bw, path_cache)
        file_envs.append((meta["name"], env, path_cache, sorted_reqs))
    print(f"  Done. {len(file_envs)} file(s) cached.", flush=True)

    from strategy.best_fit import BestFit
    from env.request import SFC as SFCcls

    t0 = time.time()
    total_teacher_success = 0
    total_teacher_seen = 0
    total_transitions = 0

    for ep in range(1, episodes + 1):
        file_name, env, path_cache, sorted_reqs = file_envs[(ep - 1) % len(file_envs)]
        env.reset()
        teacher = BestFit(env)
        transitions_count = 0
        teacher_success = 0

        z_cache = {}
        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            key = (t_s, round(req.bw, 1))
            if key not in z_cache:
                X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
                Z = vgae.encode(X, A) if len(dcs) >= 2 else np.zeros((0, LATENT_DIM), np.float32)
                z_cache[key] = (Z, dcs)

        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            key = (t_s, round(req.bw, 1))
            Z, dcs = z_cache[key]
            total_teacher_seen += 1

            if len(dcs) == 0:
                continue

            sfc = SFCcls(req)
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

                vnf_feat = np.array([vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE], dtype=np.float32)
                valid = [
                    i for i, d in enumerate(dcs)
                    if i < MAX_DCS and env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)
                ]
                if not valid:
                    continue

                buf_ll.push((Z, vnf_feat, act_idx, 1.0, Z, valid, False))
                transitions_count += 1

            success, _, _ = env.step(plan)
            if success:
                teacher_success += 1
                total_teacher_success += 1

        if len(buf_ll) >= batch:
            n_batches = min(len(buf_ll) // batch, 10)
            for _ in range(n_batches):
                ll_agent.train(buf_ll, batch)

        total_transitions += transitions_count
        if ep % 10 == 0 or ep == episodes or ep == 1:
            elapsed = time.time() - t0
            eta = (episodes - ep) * (elapsed / ep) if ep > 0 else 0
            eta_str = f"{eta / 3600:.1f}h" if eta > 3600 else (f"{eta / 60:.1f}m" if eta > 60 else f"{eta:.0f}s")
            teacher_ar = teacher_success / max(1, len(sorted_reqs))
            print(
                f"  ep {ep:>3}/{episodes}  file={file_name}"
                f"  trans={transitions_count}  teacher_acc={teacher_ar:.1%}"
                f"  buf={len(buf_ll)}  elapsed={elapsed:.1f}s  ETA={eta_str}",
                flush=True,
            )

    ll_agent.policy_net(dummy)
    ll_agent.target_net(dummy)

    os.makedirs(LL_DIR, exist_ok=True)
    out = os.path.join(LL_DIR, LL_WEIGHTS_FILE)
    np.save(out, np.array(ll_agent.policy_net.get_weights(), dtype=object), allow_pickle=True)
    if total_teacher_seen > 0:
        print(f"[Pretrain-LL] Teacher acceptance={total_teacher_success / total_teacher_seen:.1%}  transitions={total_transitions}", flush=True)
        if total_transitions < max(1_000, episodes * 50):
            print("[Pretrain-LL] Warning: transition count is low; increase --max-requests or --ll-episodes.", flush=True)
    print(f"[Pretrain-LL] Saved -> {out}", flush=True)
    return ll_agent


def main():
    parser = argparse.ArgumentParser(description="Pre-train VGAE and LL-DQN for HRL-VGAE")
    _add_pretrain_args(parser)
    args = parser.parse_args()

    train_dir = os.path.abspath(args.train_dir)
    if not os.path.isdir(train_dir):
        print(f"[Pretrain] Train dir not found: {train_dir}", flush=True)
        return

    selected = select_train_files(
        train_dir,
        profile=args.profile,
        max_files=args.max_files,
        min_servers=args.min_servers,
        min_server_ratio=args.min_server_ratio,
    )
    if not selected:
        print("[Pretrain] No training files selected.", flush=True)
        return

    print_selected_files(selected, args.max_requests)

    vgae = None
    if args.phase in ("vgae", "both"):
        vgae = pretrain_vgae(selected, epochs=args.vgae_epochs,
                             max_requests=args.max_requests, sample_mode=args.sample_mode)

    if args.phase in ("ll", "both"):
        if vgae is None:
            vgae = VGAENetwork(latent_dim=LATENT_DIM)
            vgae_path = os.path.join(VGAE_DIR, VGAE_WEIGHTS_FILE)
            if os.path.exists(vgae_path):
                vgae.load_weights(vgae_path)
            else:
                print(f"[Pretrain-LL] VGAE weights not found: {vgae_path}", flush=True)
                return
        pretrain_ll(selected, vgae, episodes=args.ll_episodes,
                    max_requests=args.max_requests, sample_mode=args.sample_mode)

    print("[Pretrain] Finished all requested phases.", flush=True)


if __name__ == "__main__":
    main()
