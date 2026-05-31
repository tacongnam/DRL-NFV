from __future__ import annotations
import os, sys, json, time, random
import numpy as np
import networkx as nx
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from models.model import VGAENetwork, ReplayBuffer
from models.placer import PlacerAgent, PressureNode
from utils.helpers import resolve_request_limit
from utils.training_logger import TrainingLogger
from data.load_data import load_env_from_json


# `get_train_files` removed in favor of callers enumerating files directly.


def print_selected_files(files: list, request_pct: int = 0):
    print(f"[Pretrain] {len(files)} file(s)", flush=True)
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        total = len(data.get("R", []))
        lim = resolve_request_limit(total, request_pct=request_pct)
        print(f"  {os.path.basename(fp)}: {min(total, lim) if lim else total}/{total}", flush=True)


def build_dc_graph(env, t_start: int, t_end: int, bw: float,
                   path_cache: dict, vnf_demand: dict = None):
    dcs = [nid for nid, n in env.network.nodes.items() if n.type == config.NODE_DC]
    n = len(dcs)
    if n == 0:
        return np.zeros((0, VGAENetwork.NODE_FEAT_DIM), np.float32), np.zeros((0, 0), np.float32), []

    cache_key = (t_start, t_end, round(bw, 1))
    if cache_key in path_cache:
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
        path_cache[cache_key] = all_len

    max_r = {k: 1.0 for k in config.RESOURCE_TYPE}
    for nd in env.network.nodes.values():
        if nd.type == config.NODE_DC and nd.cap:
            for k in config.RESOURCE_TYPE:
                max_r[k] = max(max_r[k], nd.cap[k])

    demand = vnf_demand or {k: 0.0 for k in config.RESOURCE_TYPE}

    X = np.zeros((n, VGAENetwork.NODE_FEAT_DIM), np.float32)
    A = np.zeros((n, n), np.float32)
    for i, did in enumerate(dcs):
        node = env.network.nodes[did]
        res = node.get_min_available_resource(t_start, t_end)
        cap = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
        for j, k in enumerate(config.RESOURCE_TYPE):
            X[i, j] = res[k] / max(max_r[k], 1e-6)
            slack = res[k] - demand.get(k, 0.0)
            X[i, j + 3] = min(slack / max(max_r[k], 1e-6), 1.0) if slack > 0 else 0.0
        for jj, dj in enumerate(dcs):
            if i == jj:
                A[i, jj] = 1.0
            else:
                dist = all_len.get(did, {}).get(dj)
                if dist is not None and dist > 0:
                    A[i, jj] = A[jj, i] = 1.0 / (dist + 1.0)
    return X, A, dcs


def pretrain_vgae(train_files: list, epochs: int = 60, batch: int = 16,
                  request_pct: int = 0, logger: TrainingLogger = None):
    if not train_files:
        return None
    print(f"\n{'='*50}\nVGAE Pre-training ({len(train_files)} files, {epochs} epochs)\n{'='*50}", flush=True)
    vgae = VGAENetwork(latent_dim=config.LATENT_DIM)
    buffer = ReplayBuffer(capacity=2000)
    path_cache = {}
    for fp in train_files:
        env = load_env_from_json(fp, request_pct=request_pct)
        env.reset()
        for req in env.requests:
            t_s = env._get_timeslot(req.arrival_time)
            t_e = env._get_timeslot(req.arrival_time + req.delay_max)
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache)
            if len(dcs) >= 2:
                buffer.push((X, A))
    print(f"  Collected {len(buffer)} snapshots", flush=True)
    if len(buffer) < 4:
        return None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        loss = vgae.train(buffer, epochs=1, batch=batch)
        if logger:
            logger.log_vgae_pretrain(ep, loss or 0.0)
        if ep % 20 == 0 or ep == epochs:
            print(f"  epoch {ep}/{epochs}  ({time.time()-t0:.1f}s)", flush=True)
    os.makedirs(config.VGAE_DIR, exist_ok=True)
    out = os.path.join(config.VGAE_DIR, config.VGAE_WEIGHTS_FILE)
    vgae.save_weights(out)
    print(f"[Pretrain-VGAE] Saved -> {out}", flush=True)
    return vgae


def _best_valid_dc(dcs: list, valid: list, env, vnf, t_s: int, t_e: int) -> int:
    best_idx, best_waste = valid[0], float('inf')
    for idx in valid:
        res = env.network.nodes[dcs[idx]].get_min_available_resource(t_s, t_e)
        waste = sum(res[k] - vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE)
        if waste < best_waste:
            best_waste, best_idx = waste, idx
    return best_idx


def pretrain_placer(train_files: list, vgae: VGAENetwork, episodes: int = 60,
                    batch: int = 32, request_pct: int = 0, logger: TrainingLogger = None):
    if not train_files:
        return None
    tf.keras.backend.clear_session()
    input_dim = config.LATENT_DIM * 2 + PlacerAgent.FEAT_DIM_EXTRA
    placer = PlacerAgent(latent_dim=config.LATENT_DIM, max_dcs=config.MAX_DCS, input_dim=input_dim)
    dummy = np.zeros((1, input_dim), dtype=np.float32)
    placer.policy_net(dummy)
    placer.target_net(dummy)
    placer.weight_net(dummy)
    buf = ReplayBuffer(capacity=20_000)

    file_envs = []
    for fp in train_files:
        env = load_env_from_json(fp, request_pct=request_pct)
        env.reset()
        path_cache = {}
        sorted_reqs = sorted(env.requests, key=lambda r: r.arrival_time)
        file_envs.append((os.path.basename(fp), env, path_cache, sorted_reqs))

    from strategy.best_fit import BestFit
    from env.request import SFC as SFCcls

    print(f"\n{'='*50}\nPlacer Pre-training ({len(file_envs)} files, {episodes} episodes)\n{'='*50}", flush=True)

    for ep in range(1, episodes + 1):
        epsilon = max(0.05, 0.5 * (1.0 - ep / episodes))
        _, env, path_cache, sorted_reqs = file_envs[(ep - 1) % len(file_envs)]
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
            vnf_demand = req.vnfs[0].resource if req.vnfs else {}
            X, A, dcs = build_dc_graph(env, t_s, t_e, req.bw, path_cache, vnf_demand)
            if len(dcs) < 2:
                continue
            Z = vgae.encode(X, A, deterministic=True)
            sfc = SFCcls(req)
            plan = teacher.get_placement(sfc, req.arrival_time)

            if plan is None:
                for vnf in req.vnfs:
                    vnf_feat = [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
                    valid = [i for i, d in enumerate(dcs)
                             if i < config.MAX_DCS and
                             env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)]
                    if not valid:
                        continue
                    act_idx = (random.choice(valid) if random.random() < epsilon
                               else _best_valid_dc(dcs, valid, env, vnf, t_s, t_e))
                    buf.push((Z, vnf_feat, zeros, 0.0, act_idx, -1.0, Z, valid, zeros, 0.0, False))
                continue

            max_cost = max(1.0, sum(
                max((n.get_cost(v) for n in env.network.nodes.values()
                     if n.type == config.NODE_DC and n.cost is not None
                     and n.get_cost(v) < float('inf')), default=0.0)
                for v in req.vnfs))

            prev_loc_z = zeros
            node_plan_map = {int(k.split("_")[0]): v
                             for k, v in plan.get("nodes", {}).items()}

            for k, vnf in enumerate(req.vnfs):
                node_plan = node_plan_map.get(k)
                if node_plan is None or node_plan["dc"] not in dcs:
                    continue
                act_idx = dcs.index(node_plan["dc"])
                if act_idx >= config.MAX_DCS:
                    continue
                vnf_feat = [vnf.resource.get(rk, 0.0) for rk in config.RESOURCE_TYPE]
                valid = [i for i, d in enumerate(dcs)
                         if i < config.MAX_DCS and
                         env._check_can_deploy_vnf(env.network.nodes[d], vnf, t_s, t_e)]
                if not valid:
                    continue

                chosen_node = env.network.nodes[node_plan["dc"]]
                res = chosen_node.get_min_available_resource(t_s, t_e)
                cap = chosen_node.cap or {rk: 1.0 for rk in config.RESOURCE_TYPE}
                node_press = PressureNode.compute(res, vnf.resource, cap)

                alpha, beta = placer.get_reward_weights(Z, vnf_feat, prev_loc_z, node_press)
                raw_cost = chosen_node.get_cost(vnf)
                if raw_cost == float('inf'):
                    raw_cost = max_cost
                cost_norm = min(1.0, raw_cost / max_cost)
                time_rem = max(0.0, req.end_time - req.arrival_time)
                delay_norm = 1.0 - min(1.0, time_rem / max(req.delay_max, 1e-6))
                reward = float(config.HRL_R_BASE_LL
                               + alpha * (1.0 - delay_norm)
                               - beta * cost_norm
                               - node_press)

                cur_loc_z = Z[act_idx].copy() if act_idx < len(Z) else zeros
                if k + 1 < len(req.vnfs):
                    next_vnf = req.vnfs[k + 1]
                    next_valid = [i for i, d in enumerate(dcs)
                                  if i < config.MAX_DCS and
                                  env._check_can_deploy_vnf(
                                      env.network.nodes[d], next_vnf, t_s, t_e)] or valid
                    chosen_node.use(vnf.resource, t_s, t_e + 1)
                    X_next, A_next, _ = build_dc_graph(env, t_s, t_e, req.bw, path_cache, next_vnf.resource)
                    Z_next = vgae.encode(X_next, A_next, deterministic=True)
                    chosen_node.use({rk: -vnf.resource[rk] for rk in config.RESOURCE_TYPE}, t_s, t_e + 1)
                    next_press = node_press
                else:
                    next_valid, Z_next, next_press = valid, Z, 0.0
                    cur_loc_z = zeros

                buf.push((Z, vnf_feat, prev_loc_z, node_press, act_idx, reward,
                          Z_next, next_valid, cur_loc_z, next_press,
                          k == len(req.vnfs) - 1))
                prev_loc_z = cur_loc_z

            env.step(plan)

        if len(buf) >= batch:
            for _ in range(min(len(buf) // batch, 10)):
                placer.train(buf, batch)
            if logger:
                recent = list(buf.buf)[-batch:]
                avg_r = float(np.mean([r[5] if not hasattr(r[5], '__len__') else r[5][0]
                                       for r in recent]))
                logger.log_ll_pretrain(ep, 0.0, avg_r)

        if ep == 1 or ep % 10 == 0 or ep == episodes:
            print(f"  [Placer] ep {ep}/{episodes}  buf={len(buf)}", flush=True)

    placer.policy_net(dummy)
    placer.target_net(dummy)
    placer.weight_net(dummy)
    os.makedirs(config.PLACER_DIR, exist_ok=True)
    np.save(os.path.join(config.PLACER_DIR, config.PLACER_WEIGHTS_FILE),
            np.array(placer.policy_net.get_weights(), dtype=object), allow_pickle=True)
    np.save(os.path.join(config.PLACER_DIR, config.PLACER_WEIGHT_NET_FILE),
            np.array(placer.weight_net.get_weights(), dtype=object), allow_pickle=True)
    print(f"[Pretrain-Placer] Saved -> {config.PLACER_DIR}", flush=True)
    return placer