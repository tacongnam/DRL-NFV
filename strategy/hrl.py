from __future__ import annotations

import os
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

import config
from env.env import Strategy
from env.request import SFC
from models import ReplayBuffer, VGAENetwork, HighLevelAgent, LowLevelAgent
from utils.hrl_utils import (
    LRUCache, snapshot_network, restore_network, resolve_npy_path,
    get_next_time, extract_node_plan_map
)
from utils.training_logger import TrainingLogger

class HRL_VGAE_Strategy(Strategy):
    def __init__(self, env, is_training=False, episodes=300,
                 use_ll_score=True, ll_pretrained_path=None, logger: TrainingLogger = None):
        Strategy.__init__(self, env)
        self.name         = "HRL-VGAE"
        self.is_training  = is_training
        self.episodes     = episodes
        self.use_ll_score = use_ll_score
        self.logger       = logger

        self.vgae_net = VGAENetwork(latent_dim=config.LATENT_DIM)
        self.hl_agent = HighLevelAgent(
            latent_dim=config.LATENT_DIM, use_ll_score=use_ll_score,
            input_dim=config.LATENT_DIM + HighLevelAgent.FEAT_PER_SFC)
        self.ll_agent = LowLevelAgent(
            latent_dim=config.LATENT_DIM, max_dcs=config.MAX_DCS,
            input_dim=config.LATENT_DIM * 2 + 3)

        _dummy = np.zeros((1, config.LATENT_DIM * 2 + 3), dtype=np.float32)
        self.ll_agent.policy_net(_dummy)
        self.ll_agent.target_net(_dummy)
        self.ll_agent.weight_net(_dummy)

        self.buf_HL    = ReplayBuffer(capacity=5_000)
        self.buf_LL    = ReplayBuffer(capacity=10_000)
        self.buf_Graph = ReplayBuffer(capacity=1_000)
        self._ll_traj: List[dict] = []

        self._nx_graph_cache = LRUCache(max_size=config.HRL_MAX_NX_GRAPH_CACHE)
        self._path_cache     = LRUCache(max_size=config.HRL_MAX_PATH_CACHE)
        self._graph_cache    = LRUCache(max_size=config.HRL_MAX_GRAPH_CACHE)
        self._routing_cache  = LRUCache(max_size=config.HRL_MAX_ROUTING_CACHE)
        self._max_cost_cache: dict = {}

        self._current_Z:  Optional[np.ndarray] = None
        self._current_dc: List[str]            = []

        self._dc_list = [
            nid for nid, n in env.network.nodes.items()
            if n.type == config.NODE_DC
        ]
        self._max_res = {k: 1.0 for k in config.RESOURCE_TYPE}
        for node in env.network.nodes.values():
            if node.type == config.NODE_DC and node.cap:
                for k in config.RESOURCE_TYPE:
                    self._max_res[k] = max(self._max_res[k], node.cap[k])

        self._best_fit = None

        if ll_pretrained_path:
            self._load_ll(ll_pretrained_path)

    def load_model(self, directory: str):
        self._load_ll(os.path.join(directory, config.LL_WEIGHTS_FILE))
        self._load_hl(os.path.join(directory, config.HL_WEIGHTS_FILE))
        vgae_path = os.path.join(directory, config.VGAE_WEIGHTS_FILE)
        if os.path.exists(vgae_path):
            self.vgae_net.load_weights(vgae_path)

    def _load_ll(self, path: str):
        if not path:
            return
        path = resolve_npy_path(path, config.LL_WEIGHTS_FILE)
        if not os.path.exists(path):
            print(f"[HRL] LL weights not found: {path}")
            return
        try:
            _dummy = np.zeros((1, config.LATENT_DIM * 2 + 3), dtype=np.float32)
            self.ll_agent.policy_net(_dummy)
            self.ll_agent.policy_net.set_weights(list(np.load(path, allow_pickle=True)))
            wn_path = path.replace(config.LL_WEIGHTS_FILE, config.LL_WEIGHT_NET_FILE)
            if os.path.exists(wn_path):
                self.ll_agent.weight_net.set_weights(
                    list(np.load(wn_path, allow_pickle=True)))
            print(f"[HRL] LL loaded <- {path}")
        except Exception as e:
            print(f"[HRL] LL load warning: {e}")

    def _load_hl(self, path: str):
        if not path:
            return
        path = resolve_npy_path(path, config.HL_WEIGHTS_FILE)
        if not os.path.exists(path):
            return
        try:
            _dummy = np.zeros(
                (1, config.LATENT_DIM + HighLevelAgent.FEAT_PER_SFC), dtype=np.float32)
            self.hl_agent.policy_net_ar(_dummy)
            self.hl_agent.policy_net_cost(_dummy)
            self.hl_agent.target_net_ar(_dummy)
            self.hl_agent.target_net_cost(_dummy)
            w = list(np.load(path, allow_pickle=True))
            if len(w) == 2:
                self.hl_agent.policy_net_ar.set_weights(w[0])
                self.hl_agent.target_net_ar.set_weights(w[0])
                self.hl_agent.policy_net_cost.set_weights(w[1])
                self.hl_agent.target_net_cost.set_weights(w[1])
            print(f"[HRL] HL loaded <- {path}")
        except Exception as e:
            print(f"[HRL] HL load warning: {e}")

    def _bw_pruned_graph(self, t_start: int, t_end: int, bw: float) -> nx.Graph:
        key = (t_start, t_end, round(bw, 2))
        cached = self._nx_graph_cache.get(key)
        if cached is not None:
            return cached
        G = nx.Graph()
        for nid in self.env.network.nodes:
            G.add_node(nid)
        for link in self.env.network.links:
            if link.get_available_bandwidth(t_start, t_end) >= bw:
                load = link.get_load(t_start, t_end)
                w = link.delay + config.ROUTING_BW_WEIGHT * load
                G.add_edge(link.u.name, link.v.name, weight=w, delay=link.delay)
        self._nx_graph_cache.set(key, G)
        return G

    def _get_path_lengths(self, t_start: int, t_end: int, bw: float, sources=None) -> dict:
        key = (t_start, t_end, round(bw, 2))
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        G = self._bw_pruned_graph(t_start, t_end, bw)
        lengths = {}
        for src in (sources or list(G.nodes())):
            if src in G:
                try:
                    lengths[src] = dict(
                        nx.single_source_dijkstra_path_length(G, src, weight="weight"))
                except Exception:
                    pass
        self._path_cache.set(key, lengths)
        return lengths

    def get_routing(self, u: str, v: str, t_start: int, t_end: int,
                    bw: float, Z_t=None, dc_mapping=None) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]
        rkey = (u, v, t_start, t_end, round(bw, 2))
        cached = self._routing_cache.get(rkey)
        if cached is not None:
            return cached
        G = self._bw_pruned_graph(t_start, t_end, bw)
        try:
            path = nx.shortest_path(G, u, v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            path = None
        self._routing_cache.set(rkey, path)
        return path

    def _build_dc_graph(self, t_start: int, t_end: int,
                        bw_req: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dcs = self._dc_list
        n   = len(dcs)
        if n == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 0), np.float32), []
        all_paths = self._get_path_lengths(t_start, t_end, bw_req, sources=dcs)
        X = np.zeros((n, 3), np.float32)
        A = np.zeros((n, n), np.float32)
        for i, dc_id in enumerate(dcs):
            res  = self.env.network.nodes[dc_id].get_min_available_resource(t_start, t_end)
            X[i] = [res[k] / self._max_res[k] for k in config.RESOURCE_TYPE]
        for i, dc_i in enumerate(dcs):
            A[i, i] = 1.0
            for j in range(i + 1, n):
                dc_j  = dcs[j]
                dist  = all_paths.get(dc_i, {}).get(dc_j)
                if dist is not None and dist > 0:
                    w = 1.0 / (dist + 1.0)
                    A[i, j] = A[j, i] = w
        return X, A, dcs

    def _get_z(self, t_start, t_end, bw_req):
        key = (t_start, t_end, round(bw_req, 1))
        cached = self._graph_cache.get(key)
        if cached is None:
            X, A, dcs = self._build_dc_graph(t_start, t_end, bw_req)
            Z = self.vgae_net.encode(X, A, deterministic=not self.is_training)
            self._graph_cache.set(key, (Z, dcs, X, A))
        else:
            Z, dcs, X, A = cached
        self._current_Z  = Z
        self._current_dc = dcs
        return Z, dcs

    def _clear_caches(self):
        self._nx_graph_cache.clear()
        self._path_cache.clear()
        self._routing_cache.clear()
        self._graph_cache.clear()

    def get_placement(self, sfc: SFC, current_time: float,
                      Z_t=None, dc_mapping=None, epsilon=0.0) -> Optional[Dict]:
        self._ll_traj = []
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

        if Z_t is None or dc_mapping is None:
            Z_t, dc_mapping = self._get_z(t_start, t_end, sfc.request.bw)
        if not dc_mapping:
            return None

        node_placements, vnf_timeslots = [], []
        link_paths, link_timeslots = [], []
        prev_dc = sfc.request.start_node
        loc_z   = np.zeros(config.LATENT_DIM, dtype=np.float32)

        for vnf in sfc.request.vnfs:
            cand = [str(x) for x in vnf.get_dcs()]
            if '-1' in cand or not cand:
                cand = dc_mapping
            else:
                cand = [d for d in cand if d in dc_mapping]

            valid_indices = [
                idx for idx, dc_id in enumerate(dc_mapping)
                if dc_id in cand and idx < config.MAX_DCS
                and self.env._check_can_deploy_vnf(
                    self.env.network.nodes[dc_id], vnf, t_start, t_end)
            ]
            if not valid_indices:
                return None

            vnf_feat   = [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
            action_idx = self.ll_agent.act(Z_t, vnf_feat, valid_indices, epsilon, loc_z)
            chosen_dc  = dc_mapping[action_idx]

            path = self.get_routing(prev_dc, chosen_dc, t_start, t_end, sfc.request.bw)
            if path is None:
                return None

            edges = list(zip(path, path[1:]))
            total_delay = sum(
                link.delay for link in self.env.network.links
                if (link.u.name, link.v.name) in edges
                or (link.v.name, link.u.name) in edges
            )
            if total_delay > sfc.request.delay_max:
                return None

            self._ll_traj.append({
                "Z_t":        Z_t,
                "Z_cache_key": (t_start, t_end, round(sfc.request.bw, 1)),
                "vnf_feat":   vnf_feat,
                "loc_z":      loc_z.copy(),
                "action_idx": action_idx,
                "valid_mask": valid_indices,
            })

            if action_idx < len(Z_t):
                loc_z = Z_t[action_idx].copy()

            node_placements.append(chosen_dc)
            vnf_timeslots.append((t_start, t_end))
            link_paths.append(path)
            link_timeslots.append((t_start, t_end))
            prev_dc = chosen_dc

        final_path = self.get_routing(
            prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None
        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))
        return self.build_placement_plan(
            node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)

    def _get_best_fit(self):
        if self._best_fit is None:
            from strategy.best_fit import BestFit
            self._best_fit = BestFit(self.env)
        return self._best_fit

    def _compute_ll_reward(self, env_rewards: list, sfc: SFC,
                           current_time: float, Z_t: np.ndarray, vnf_feat: list) -> float:
        alpha, beta = self.ll_agent.get_reward_weights(Z_t, vnf_feat)
        alpha = np.clip(alpha, *config.HRL_ALPHA_CLAMP)
        beta  = np.clip(beta,  *config.HRL_BETA_CLAMP)
        time_rem  = max(0.0, sfc.request.end_time - current_time)
        tMax      = max(sfc.request.delay_max, 1e-6)
        raw_cost  = abs(env_rewards[1]) if len(env_rewards) > 1 else 0.0
        cost_norm = min(1.0, raw_cost / max(self._estimate_max_cost(sfc), 1e-6))
        return config.HRL_R_BASE_LL + alpha * (time_rem / tMax) - beta * cost_norm

    def _estimate_max_cost(self, sfc: SFC) -> float:
        key = tuple(id(v) for v in sfc.request.vnfs)
        if key in self._max_cost_cache:
            return self._max_cost_cache[key]
        max_cost = 0.0
        for vnf in sfc.request.vnfs:
            vid = id(vnf)
            if vid not in self._max_cost_cache:
                self._max_cost_cache[vid] = max(
                    (n.get_cost(vnf) for n in self.env.network.nodes.values()
                     if n.type == config.NODE_DC and n.cost is not None
                     and n.get_cost(vnf) < float("inf")),
                    default=0.0)
            max_cost += self._max_cost_cache.get(vid, 0.0)
        result = max(1.0, max_cost)
        self._max_cost_cache[key] = result
        return result

    def _execute_plan(self, plan: Optional[Dict], selected_sfc: SFC,
                      t: float, snap: dict) -> Tuple[bool, list, Optional[float], Optional[Dict]]:
        if plan is not None:
            success, rewards, score = self.env.step(plan)
            if success:
                return success, rewards, score, plan
            restore_network(self.env.network, snap)

        fallback = self._get_best_fit().get_placement(selected_sfc, t)
        if fallback is not None:
            success, rewards, score = self.env.step(fallback)
            if success:
                return success, rewards, score, fallback

        return False, [-1.0, 0.0], None, None

    def _rebuild_ll_traj_from_plan(self, plan, sfc, t, Z_t, dc_mapping):
        node_plan_map = extract_node_plan_map(plan)
        t_start = self.env._get_timeslot(t)
        t_end   = self.env._get_timeslot(sfc.request.end_time)
        prev_loc_z = np.zeros(config.LATENT_DIM, dtype=np.float32)
        for i, vnf in enumerate(sfc.request.vnfs):
            node_plan = node_plan_map.get(i)
            if node_plan is None:
                continue
            dc = node_plan["dc"]
            if dc not in dc_mapping:
                continue
            act_idx = dc_mapping.index(dc)
            if act_idx >= config.MAX_DCS:
                continue
            cand = [str(x) for x in vnf.get_dcs()]
            if '-1' in cand or not cand:
                cand = dc_mapping
            valid_indices = [
                idx for idx, dc_id in enumerate(dc_mapping)
                if dc_id in cand and idx < config.MAX_DCS
                and self.env._check_can_deploy_vnf(
                    self.env.network.nodes[dc_id], vnf, t_start, t_end)
            ]
            self._ll_traj.append({
                "Z_t":        Z_t,
                "vnf_feat":   [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE],
                "loc_z":      prev_loc_z,
                "action_idx": act_idx,
                "valid_mask": valid_indices,
            })
            prev_loc_z = Z_t[act_idx].copy() if act_idx < len(Z_t) else prev_loc_z

    def _calc_node_cost(self, plan, rewards):
        cost = sum(
            self.env.network.nodes[v["dc"]].get_cost(self.env.vnfs[v["vnf_name"]])
            for v in plan.get("nodes", {}).values()
            if v["dc"] in self.env.network.nodes and v.get("vnf_name") in self.env.vnfs
            and self.env.network.nodes[v["dc"]].get_cost(self.env.vnfs[v["vnf_name"]]) < float("inf")
        )
        if cost == float("inf") or cost < 0:
            return abs(rewards[1]) if len(rewards) > 1 else 0.0
        return cost

    def train(self) -> dict:
        total_steps = 0
        total_steps_planned = self.episodes * len(self.env.requests)
        
        # Tracking for logging
        hl_losses_ar_batch = []
        hl_losses_cost_batch = []
        ll_losses_batch = []
        ll_weight_losses_batch = []
        vgae_losses_batch = []

        ep_accepted = ep_rejected = 0
        acc_rate = 0.0
        for ep in range(1, self.episodes + 1):
            self.env.reset()
            self._clear_caches()
            self._best_fit = None

            pending = sorted(
                [SFC(r) for r in self.env.requests],
                key=lambda s: s.request.arrival_time)
            queue: List[SFC] = []
            ep_accepted = ep_rejected = 0
            t = pending[0].request.arrival_time if pending else 0.0

            while pending or queue:
                if not queue and pending:
                    t = pending[0].request.arrival_time

                while pending and pending[0].request.arrival_time <= t:
                    queue.append(pending.pop(0))

                active  = [s for s in queue if t <= s.request.end_time]
                ep_rejected += len(queue) - len(active)
                queue = active
                if not queue:
                    if pending:
                        t = pending[0].request.arrival_time
                    continue

                total_steps += 1
                progress = total_steps / max(1, total_steps_planned)
                epsilon  = max(0.05, 0.9 - progress * 1.7)

                bw_req       = max(s.request.bw for s in queue)
                t_start_rough = self.env._get_timeslot(t)
                t_end_rough   = t_start_rough + max(
                    int(max(s.request.delay_max for s in queue) / config.TIMESTEP), 10)
                Z_t, _ = self._get_z(t_start_rough, t_end_rough, bw_req)

                sfc_feats_before = self.hl_agent.extract_sfc_features(queue, Z_t, self.ll_agent)
                sfc_idx      = self.hl_agent.act(Z_t, queue, epsilon, self.ll_agent)
                selected_sfc = queue.pop(sfc_idx)

                self.env.t = t
                t_start = self.env._get_timeslot(t)
                t_end   = self.env._get_timeslot(selected_sfc.request.end_time)
                Z_t, dc_mapping = self._get_z(t_start, t_end, selected_sfc.request.bw)

                cached = self._graph_cache.get((t_start, t_end, round(selected_sfc.request.bw, 1)))
                X, A = (cached[2], cached[3]) if cached else (None, None)

                snap = snapshot_network(self.env.network)

                use_greedy   = np.random.random() < max(0.05, 1.0 - progress * 0.95)
                R_LL_override = None
                if use_greedy:
                    plan = self._get_best_fit().get_placement(selected_sfc, t)
                    self._ll_traj = []
                    R_LL_override = 1.5
                    if plan is not None:
                        self._rebuild_ll_traj_from_plan(plan, selected_sfc, t, Z_t, dc_mapping)
                else:
                    plan = self.get_placement(selected_sfc, t, Z_t, dc_mapping, epsilon)
                    if plan is None:
                        R_LL_override = 1.0

                success, rewards, score, plan = self._execute_plan(plan, selected_sfc, t, snap)

                if success:
                    ep_accepted += 1
                    raw_cost   = abs(rewards[1]) if len(rewards) > 1 else 0.0
                    cost_norm  = min(1.0, raw_cost / max(self._estimate_max_cost(selected_sfc), 1e-6))
                    time_ratio = min(1.0, (t - selected_sfc.request.arrival_time)
                                     / max(selected_sfc.request.delay_max, 1e-6))
                    R_HL = [config.HRL_BASE_REWARD + 1.0 - time_ratio, -cost_norm]
                    vnf_f = ([selected_sfc.request.vnfs[0].resource.get(k, 0.0)
                               for k in config.RESOURCE_TYPE]
                             if selected_sfc.request.vnfs else [0.0, 0.0, 0.0])
                    if R_LL_override is None:
                        R_LL = self._compute_ll_reward(rewards, selected_sfc, t, Z_t, vnf_f)
                    else:
                        R_LL = R_LL_override
                else:
                    ep_rejected += 1
                    restore_network(self.env.network, snap)
                    R_HL = [-(1.0 + len(queue) / max(self.hl_agent.max_queue, 1)), 0.0]
                    R_LL = -config.HRL_PENALTY_DROP

                self._clear_caches()
                t = max(t, get_next_time(pending, t)) if not queue else t
                is_done = not pending and not queue

                Z_mean = Z_t.mean(axis=0, keepdims=True)
                sfc_feats_next = (self.hl_agent.extract_sfc_features(queue, Z_t, self.ll_agent)
                                  if queue else sfc_feats_before)
                self.buf_HL.push((
                    Z_mean, sfc_feats_before, sfc_idx,
                    R_HL, Z_mean, sfc_feats_next, is_done))

                for i, step in enumerate(self._ll_traj):
                    nxt_mask  = (self._ll_traj[i+1]["valid_mask"]
                                 if i+1 < len(self._ll_traj) else [])
                    loc_z_next = (self._ll_traj[i+1]["loc_z"]
                                  if i+1 < len(self._ll_traj)
                                  else np.zeros(config.LATENT_DIM, dtype=np.float32))
                    self.buf_LL.push((
                        step["Z_t"], list(step["vnf_feat"]), step["loc_z"],
                        step["action_idx"], R_LL,
                        Z_t, nxt_mask, loc_z_next, is_done))

                if X is not None:
                    self.buf_Graph.push((X, A))

                # LL-Agent Training
                if total_steps % 4 == 0 and len(self.buf_LL) >= config.HRL_BATCH_SIZE:
                    self.ll_agent.train(self.buf_LL, config.HRL_BATCH_SIZE)
                    if self.logger:
                        # Approximate loss from buffer
                        ll_losses_batch.append(np.mean([abs(r[4]) for r in list(self.buf_LL.buf)[-config.HRL_BATCH_SIZE:]]))
                        ll_weight_losses_batch.append(np.mean(ll_losses_batch[-10:]) if ll_losses_batch else 0.0)

                # HL-Agent Training
                if total_steps % 8 == 0 and len(self.buf_HL) >= config.HRL_BATCH_SIZE:
                    self.hl_agent.train(self.buf_HL, config.HRL_BATCH_SIZE)
                    if self.logger:
                        hl_losses_ar_batch.append(np.mean([abs(r[3][0]) for r in list(self.buf_HL.buf)[-config.HRL_BATCH_SIZE:]]))
                        hl_losses_cost_batch.append(np.mean([abs(r[3][1]) for r in list(self.buf_HL.buf)[-config.HRL_BATCH_SIZE:]]))

                # Target Network Sync
                if total_steps % config.HRL_TARGET_SYNC == 0:
                    self.ll_agent.update_target_network()
                    self.hl_agent.update_target_networks()

                # VGAE Online Training
                if total_steps % config.HRL_VGAE_TRAIN_FREQ == 0 and len(self.buf_Graph) >= 4:
                    vgae_loss = self.vgae_net.train(self.buf_Graph, epochs=config.HRL_VGAE_EPOCHS)
                    if self.logger and vgae_loss is not None:
                        vgae_losses_batch.append(vgae_loss)
                        self.logger.log_vgae_online_train(total_steps, np.mean(vgae_losses_batch[-10:]))

                # Log step-wise metrics
                if self.logger:
                    if hl_losses_ar_batch:
                        self.logger.log_hl_train_step(total_steps, 
                                                     np.mean(hl_losses_ar_batch[-5:]),
                                                     np.mean(hl_losses_cost_batch[-5:]) if hl_losses_cost_batch else 0.0)
                    if ll_losses_batch:
                        self.logger.log_ll_train_step(total_steps,
                                                     np.mean(ll_losses_batch[-5:]),
                                                     np.mean(ll_weight_losses_batch[-5:]) if ll_weight_losses_batch else 0.0)

            # Log episode metrics
            total_ep = ep_accepted + ep_rejected
            acc_rate = ep_accepted / max(1, total_ep)
            
            if self.logger:
                self.logger.log_episode(ep, acc_rate, [ep_accepted, ep_rejected])
            
            if ep % 10 == 0 or ep == self.episodes:
                print(f"[HRL] Episode {ep}/{self.episodes}  Acceptance: {acc_rate:.3f}", flush=True)

        self.env.stats.update({
            "accepted_requests": ep_accepted,
            "rejected_requests": ep_rejected,
            "acceptance_ratio":  acc_rate,
            "algorithm_name":    self.name,
        })
        return self.env.stats

    def run_simulation_eval(self) -> dict:
        self.env.reset()
        self._clear_caches()
        self._best_fit = None

        pending = sorted([SFC(r) for r in self.env.requests],
                         key=lambda s: s.request.arrival_time)
        queue: List[SFC] = []
        accepted = rejected = 0
        total_node_cost = 0.0
        t = pending[0].request.arrival_time if pending else 0.0

        while pending or queue:
            if not queue and pending:
                t = pending[0].request.arrival_time
            while pending and pending[0].request.arrival_time <= t:
                queue.append(pending.pop(0))

            active = [s for s in queue if t <= s.request.end_time]
            rejected += len(queue) - len(active)
            queue = active
            if not queue:
                if pending:
                    t = pending[0].request.arrival_time
                continue

            bw_req      = max(s.request.bw for s in queue)
            t_start     = self.env._get_timeslot(t)
            t_end_rough = t_start + max(
                int(max(s.request.delay_max for s in queue) / config.TIMESTEP), 10)
            Z_t, _ = self._get_z(t_start, t_end_rough, bw_req)

            sfc_idx      = self.hl_agent.act(Z_t, queue, 0.0, self.ll_agent)
            selected_sfc = queue.pop(sfc_idx)

            t_end = self.env._get_timeslot(selected_sfc.request.end_time)
            Z_t, dc_mapping = self._get_z(t_start, t_end, selected_sfc.request.bw)
            self.env.t = t

            snap = snapshot_network(self.env.network)
            plan = self.get_placement(selected_sfc, t, Z_t, dc_mapping, 0.0)
            success, rewards, _, plan = self._execute_plan(plan, selected_sfc, t, snap)

            if success and plan:
                accepted += 1
                total_node_cost += self._calc_node_cost(plan, rewards)
                self.env.stats["total_delay"] += (
                    selected_sfc.request.end_time - selected_sfc.request.arrival_time)
            else:
                rejected += 1

            self._clear_caches()
            t = max(t, min((s.request.arrival_time for s in pending), default=t)) \
                if not queue else t

        total = accepted + rejected
        self.env.stats.update({
            "accepted_requests": accepted,
            "rejected_requests": rejected,
            "total_requests":    total,
            "total_cost":        total_node_cost,
            "acceptance_ratio":  accepted / total if total > 0 else 0.0,
            "average_cost":      total_node_cost / accepted if accepted > 0 else 0.0,
            "algorithm_name":    self.name,
        })
        return self.env.stats

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        try:
            np.save(os.path.join(directory, config.LL_WEIGHTS_FILE),
                    np.array(self.ll_agent.policy_net.get_weights(), dtype=object),
                    allow_pickle=True)
            np.save(os.path.join(directory, config.LL_WEIGHT_NET_FILE),
                    np.array(self.ll_agent.weight_net.get_weights(), dtype=object),
                    allow_pickle=True)
            np.save(os.path.join(directory, config.HL_WEIGHTS_FILE),
                    np.array([self.hl_agent.policy_net_ar.get_weights(),
                              self.hl_agent.policy_net_cost.get_weights()], dtype=object),
                    allow_pickle=True)
            self.vgae_net.save_weights(os.path.join(directory, config.VGAE_WEIGHTS_FILE))
            print(f"[HRL] Models saved → {directory}")
        except Exception as e:
            print(f"[HRL] Save warning: {e}")
