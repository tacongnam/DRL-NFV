import os, sys, json, time, random, numpy as np, networkx as nx, tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from env import Env
from models import VGAENetwork, ReplayBuffer, LowLevelAgent
from utils.helpers import resolve_request_limit
from data.load_data import load_env_from_json

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

def build_dc_graph(env: Env, t_start: int, t_end: int, bw: float, path_cache: dict = None):
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
                    A[j, i] = 1.0 / (dist + 1.0)
    return X, A, dcs

def pretrain_vgae(train_files: list, epochs: int = 200, batch: int = 16, request_pct: int = 0):
    if not train_files:
        print("[Pretrain-VGAE] No JSON files selected", flush=True)
        return None

    print(f"\n{'=' * 50}", flush=True)
    print(f"PHASE 1: VGAE Pre-training  ({len(train_files)} files, {epochs} epochs)", flush=True)
    print(f"{'=' * 50}", flush=True)

    vgae = VGAENetwork(latent_dim=config.LATENT_DIM)
    buffer = ReplayBuffer(capacity=2000)
    path_cache = {}
    per_file_snapshots = {}

    print("Collecting graph snapshots ...", flush=True)
    for fp in train_files:
        env = load_env_from_json(fp, request_pct=request_pct)
        env.reset()
        local_count = 0
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) >= 2:
                buffer.push((X, A))
                local_count += 1
        per_file_snapshots[os.path.basename(fp)] = local_count
    print(f"  Collected {len(buffer)} graph snapshots", flush=True)
    for name, count in per_file_snapshots.items():
        print(f"    {name}: {count}", flush=True)

    if len(buffer) < 4:
        print("[Pretrain-VGAE] Too few snapshots, skipping", flush=True)
        return None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        vgae.train(buffer, epochs=1, batch=batch)
        if ep % 50 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}  ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(config.VGAE_DIR, exist_ok=True)
    out = os.path.join(config.VGAE_DIR, config.VGAE_WEIGHTS_FILE)
    vgae.save_weights(out)
    print(f"[Pretrain-VGAE] Saved -> {out}", flush=True)
    return vgae

def _get_node_plan_map(plan: dict) -> dict:
    result = {}
    for vnf_key, vplan in plan.get("nodes", {}).items():
        idx = int(vnf_key.split("_")[0])
        result[idx] = vplan
    return result

def _best_valid_dc(dcs: list, valid: list, env: Env, vnf, t_s: int, t_e: int) -> int:
    best_idx   = valid[0]
    best_waste = float('inf')
    for idx in valid:
        dc_name = dcs[idx]
        node    = env.network.nodes[dc_name]
        res     = node.get_min_available_resource(t_s, t_e)
        waste   = sum(res[k] - vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE)
        if waste < best_waste:
            best_waste = waste
            best_idx   = idx
    return best_idx


def pretrain_ll(train_files: list, vgae: VGAENetwork, episodes: int = 200, batch: int = 32, request_pct: int = 0):
    if not train_files:
        return None

    tf.keras.backend.clear_session()

    ll_agent = LowLevelAgent(latent_dim=config.LATENT_DIM, max_dcs=config.MAX_DCS, input_dim=config.LATENT_DIM * 2 + 3)
    dummy = np.zeros((1, config.LATENT_DIM * 2 + 3), dtype=np.float32)
    ll_agent.policy_net(dummy)
    ll_agent.target_net(dummy)
    ll_agent.weight_net(dummy)
    buf_ll = ReplayBuffer(capacity=20_000)
    file_envs = []

    for fp in train_files:
        env = load_env_from_json(fp, request_pct=request_pct)
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

    PRETRAIN_EPSILON_START = 0.5
    PRETRAIN_EPSILON_END   = 0.05

    for ep in range(1, episodes + 1):
        pretrain_epsilon = max(
            PRETRAIN_EPSILON_END,
            PRETRAIN_EPSILON_START * (1.0 - ep / episodes),
        )

        file_name, env, path_cache, sorted_reqs = file_envs[(ep - 1) % len(file_envs)]
        env.reset()
        teacher = BestFit(env)
        zeros = np.zeros(config.LATENT_DIM, dtype=np.float32)

        max_r = {k: 1.0 for k in config.RESOURCE_TYPE}
        for nd in env.network.nodes.values():
            if nd.type == config.NODE_DC and nd.cap:
                for k in config.RESOURCE_TYPE:
                    max_r[k] = max(max_r[k], nd.cap[k])

        for req in sorted_reqs:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.end_time)

            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) < 2:
                continue
            Z = vgae.encode(X, A, deterministic=True)

            sfc  = SFCcls(req)
            plan = teacher.get_placement(sfc, req.arrival_time)

            if plan is None:    # Not success
                for vnf in req.vnfs:
                    vnf_feat = np.array([vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE], dtype=np.float32)
                    valid = [
                        i for i, d in enumerate(dcs)
                        if i < config.MAX_DCS and env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)
                    ]
                    if not valid:
                        continue

                    if random.random() < pretrain_epsilon:
                        best_idx = random.choice(valid)
                    else:
                        best_idx = _best_valid_dc(dcs, valid, env, vnf, t_s, t_e)
                    buf_ll.push((Z, vnf_feat, zeros, best_idx, -1.0, Z, valid, zeros, False))
                continue

            # Success
            max_cost = 0.0
            for v in req.vnfs:
                costs = [n.get_cost(v) for n in env.network.nodes.values() if n.type == config.NODE_DC and n.cost is not None]
                finite = [c for c in costs if c < float('inf')]
                if finite:
                    max_cost += max(finite)
            max_cost = max(1.0, max_cost)

            prev_loc_z = zeros
            node_plan_map = _get_node_plan_map(plan)

            for k, vnf in enumerate(req.vnfs):
                node_plan = node_plan_map.get(k)
                if node_plan is None:
                    continue
                greedy_dc = node_plan["dc"]
                if greedy_dc not in dcs:
                    continue
                act_idx = dcs.index(greedy_dc)
                if act_idx >= config.MAX_DCS:
                    continue

                vnf_feat = np.array([vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE], dtype=np.float32)
                valid = [i for i, d in enumerate(dcs)
                        if i < config.MAX_DCS and env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)]
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

                cur_loc_z = Z[act_idx].copy() if act_idx < len(Z) else np.zeros(config.LATENT_DIM, np.float32)
                if k + 1 < len(req.vnfs):
                    next_vnf   = req.vnfs[k + 1]
                    next_valid = [i for i, d in enumerate(dcs)
                                if i < config.MAX_DCS and env._check_can_deploy_vnf(
                                    env.network.nodes[d], next_vnf, t_s, t_e)] or valid
                    next_loc_z = cur_loc_z

                    chosen_node.use(vnf.resource, t_s, t_e + 1)
                    X_next, A_next, _ = build_dc_graph(env, t_s, t_e, req.bw, {})
                    Z_next = vgae.encode(X_next, A_next, deterministic=True)
                    chosen_node.use({k_: -vnf.resource[k_] for k_ in config.RESOURCE_TYPE}, t_s, t_e + 1)
                else:
                    next_valid = valid
                    next_loc_z = np.zeros(config.LATENT_DIM, np.float32)
                    Z_next = Z

                buf_ll.push((Z, vnf_feat, prev_loc_z,
                            act_idx, reward,
                            Z_next, next_valid, next_loc_z,
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
    ll_agent.weight_net(dummy)

    os.makedirs(config.LL_DIR, exist_ok=True)
    
    out_policy = os.path.join(config.LL_DIR, config.LL_WEIGHTS_FILE)
    np.save(out_policy, np.array(ll_agent.policy_net.get_weights(), dtype=object), allow_pickle=True)

    out_weight_net = os.path.join(config.LL_DIR, config.LL_WEIGHT_NET_FILE)
    np.save(out_weight_net, np.array(ll_agent.weight_net.get_weights(), dtype=object), allow_pickle=True)

    print(f"[Pretrain-LL] policy_net saved  -> {out_policy}", flush=True)
    print(f"[Pretrain-LL] weight_net saved  -> {out_weight_net}", flush=True)

    return ll_agent