import os, sys, json, argparse, time, math
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
DEFAULT_VGAE_EPOCHS = 60
DEFAULT_LL_EPISODES = 60
DEFAULT_REQUEST_PCT = 0


def sample_requests(req_rows: list, request_pct: int = 0) -> list:
    req_limit = resolve_request_limit(len(req_rows), request_pct=request_pct)
    if req_limit is None or req_limit <= 0 or len(req_rows) <= req_limit:
        return req_rows
    idxs = np.linspace(0, len(req_rows) - 1, num=req_limit, dtype=int)
    return [req_rows[i] for i in idxs]


def resolve_request_limit(total_requests: int, request_pct: int = 0) -> int | None:
    if request_pct is None or request_pct <= 0:
        return None
    return max(1, math.ceil(total_requests * request_pct / 100.0))


def load_env(path: str, request_pct: int = 0) -> Env:
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
    req_rows = sample_requests(req_rows, request_pct=request_pct)

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


def get_train_files(train_dir: str) -> list:
    return sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".json")])


def print_selected_files(files: list, request_pct: int = 0):
    print(f"[Pretrain] Selected {len(files)} file(s)", flush=True)
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        total_requests = len(data.get("R", []))
        req_limit = resolve_request_limit(total_requests, request_pct=request_pct)
        req_label = total_requests if req_limit is None else min(total_requests, req_limit)
        print(f"  - {os.path.basename(fp)}: req={req_label}/{total_requests}", flush=True)


def _add_pretrain_args(parser: argparse.ArgumentParser):
    parser.add_argument("--phase", type=str, default="both", choices=["vgae", "ll", "both"])
    parser.add_argument("--train-dir", type=str, default="data/train")
    parser.add_argument("--request-pct", type=int, default=DEFAULT_REQUEST_PCT)
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
                  request_pct: int = 0):
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
    for fp in train_files:
        env = load_env(fp, request_pct=request_pct)
        env.reset()
        local_count = 0
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) >= 2:
                buf.push((X, A))
                local_count += 1
        per_file_snapshots[os.path.basename(fp)] = local_count
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


def pretrain_ll(train_files: list, vgae: VGAENetwork, episodes: int = 200, batch: int = 32, request_pct: int = 0):
    if not train_files:
        return None
    
    import tensorflow as tf
    tf.keras.backend.clear_session()

    ll_agent = LowLevelAgent(latent_dim=LATENT_DIM, max_dcs=MAX_DCS, input_dim=LATENT_DIM * 2 + 3)
    dummy = np.zeros((1, LATENT_DIM * 2 + 3), dtype=np.float32)
    ll_agent.policy_net(dummy)
    ll_agent.target_net(dummy)
    buf_ll = ReplayBuffer(capacity=20_000)
    file_envs = []

    for fp in train_files:
        env = load_env(fp, request_pct=request_pct)
        env.reset()
        path_cache = {}
        sorted_reqs = sorted(env.requests, key=lambda r: r.arrival_time)

        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            build_dc_graph(env, t_s, t_e, req.bw, path_cache)
        file_envs.append((os.path.basename(fp), env, path_cache, sorted_reqs))

    from strategy.best_fit import BestFit
    from env.request import SFC as SFCcls

    for ep in range(1, episodes + 1):
        file_name, env, path_cache, sorted_reqs = file_envs[(ep - 1) % len(file_envs)]
        env.reset()
        teacher = BestFit(env)
        z_cache = {}

        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            key = (t_s, t_e, round(req.bw, 1))

            if key not in z_cache:
                X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
                Z = vgae.encode(X, A) if len(dcs) >= 2 else np.zeros((0, LATENT_DIM), np.float32)
                z_cache[key] = (Z, dcs)

        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)
            key = (t_s, t_e, round(req.bw, 1))
            Z, dcs = z_cache[key]

            if len(dcs) == 0:
                continue

            sfc = SFCcls(req)
            plan = teacher.get_placement(sfc, req.arrival_time)

            if plan is None:
                for vnf in req.vnfs:
                    vnf_feat = np.array([vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE], dtype=np.float32)
                    valid = [i for i, d in enumerate(dcs) if i < MAX_DCS and env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)]

                    if not valid:
                        continue

                    best_idx = valid[0]
                    buf_ll.push((Z, vnf_feat, best_idx, -1.0, Z, valid, False))
                continue

            max_cost = 0.0

            for v in req.vnfs:
                costs = [n.get_cost(v) for n in env.network.nodes.values() if n.type == config.NODE_DC and n.cost is not None]
                finite = [c for c in costs if c < float('inf')]
                if finite:
                    max_cost += max(finite)

            max_cost = max(1.0, max_cost)
            prev_loc_z = np.zeros(LATENT_DIM, dtype=np.float32)
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
                valid = [i for i, d in enumerate(dcs)
                        if i < MAX_DCS and env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)]
                if not valid:
                    continue
                chosen_node = env.network.nodes[greedy_dc]
                alpha, beta = ll_agent.get_reward_weights(Z, vnf_feat, prev_loc_z)
                time_rem  = max(0.0, req.end_time - req.arrival_time)
                tMax      = max(req.delay_max, 1e-6)
                raw_cost  = chosen_node.get_cost(vnf) if chosen_node.cost is not None else 0.0
                if raw_cost == float('inf'):
                    raw_cost = max_cost
                cost_norm = min(1.0, raw_cost / max_cost)
                reward    = float(1.0 + alpha * (time_rem / tMax) - beta * cost_norm)
                # loc_z tiếp theo = latent vector của DC vừa chọn
                cur_loc_z = Z[act_idx].copy() if act_idx < len(Z) else np.zeros(LATENT_DIM, np.float32)
                if k + 1 < len(req.vnfs):
                    next_vnf   = req.vnfs[k + 1]
                    next_valid = [i for i, d in enumerate(dcs)
                                if i < MAX_DCS and env._check_can_deploy_vnf(
                                    env.network.nodes[d], next_vnf, t_s, t_e)] or valid
                    next_loc_z = cur_loc_z
                else:
                    next_valid = valid
                    next_loc_z = np.zeros(LATENT_DIM, np.float32)
                buf_ll.push((Z, vnf_feat, prev_loc_z,
                            act_idx, reward,
                            Z, next_valid, next_loc_z,
                            k == len(req.vnfs) - 1))
                prev_loc_z = cur_loc_z
            env.step(plan)

        if len(buf_ll) >= batch:
            n_batches = min(len(buf_ll) // batch, 10)
            for _ in range(n_batches):
                ll_agent.train(buf_ll, batch)

        if ep == 1 or ep % 10 == 0 or ep == episodes:
            print(f"  [LL] episode {ep}/{episodes}  buf={len(buf_ll)}", flush=True)

    ll_agent.policy_net(dummy)
    ll_agent.target_net(dummy)
    os.makedirs(LL_DIR, exist_ok=True)
    out = os.path.join(LL_DIR, LL_WEIGHTS_FILE)
    np.save(out, np.array(ll_agent.policy_net.get_weights(), dtype=object), allow_pickle=True)
    return ll_agent