from __future__ import annotations

import os, copy, math, time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

import config
from env.env import Strategy
from env.request import SFC
from models.model import ReplayBuffer, VGAENetwork, HighLevelAgent, LowLevelAgent

BASE_AR_REWARD  = 1.0
PENALTY_DROP    = 0.5
R_BASE_LL       = 1.0
BATCH_SIZE      = 32
TARGET_SYNC     = 100
VGAE_TRAIN_FREQ = 100
LATENT_DIM      = 8
MAX_DCS         = 60

_LL_WEIGHTS_FILE   = "ll_dqn_weights.npy"
_HL_WEIGHTS_FILE   = "hl_pmdrl_weights.npy"
_VGAE_WEIGHTS_FILE = "vgae_weights.npy"


class HRL_VGAE_Strategy(Strategy):

    def __init__(self, env,
                 is_training:        bool  = False,
                 episodes:           int   = 300,
                 use_ll_score:       bool  = True,
                 ll_pretrained_path: Optional[str] = None):
        super().__init__(env)
        self.name            = "HRL-VGAE"
        self.is_training     = is_training
        self.episodes        = episodes
        self.use_ll_score    = use_ll_score

        self.vgae_net = VGAENetwork(latent_dim=LATENT_DIM)
        self.hl_agent = HighLevelAgent(latent_dim=LATENT_DIM,
                                       use_ll_score=use_ll_score,
                                       input_dim=LATENT_DIM + HighLevelAgent.FEAT_PER_SFC)
        self.ll_agent = LowLevelAgent(latent_dim=LATENT_DIM,
                                      max_dcs=MAX_DCS,
                                      input_dim=LATENT_DIM + 3)

        dummy = np.zeros((1, LATENT_DIM + 3), dtype=np.float32)
        self.ll_agent.policy_net(dummy)
        self.ll_agent.target_net(dummy)

        self.buf_HL    = ReplayBuffer(capacity=5_000)
        self.buf_LL    = ReplayBuffer(capacity=10_000)
        self.buf_Graph = ReplayBuffer(capacity=1_000)

        self._total_steps_global = 0
        self._ll_traj: List[dict] = []

        self._graph_cache:    dict = {}
        self._nx_graph_cache: dict = {}
        self._path_cache:     dict = {}
        self._max_cost_cache: dict = {}

        self._dc_list: List[str] = [
            nid for nid, n in env.network.nodes.items()
            if n.type == config.NODE_DC
        ]
        self._max_res: dict = {k: 1.0 for k in config.RESOURCE_TYPE}
        for node in env.network.nodes.values():
            if node.type == config.NODE_DC and node.cap:
                for k in config.RESOURCE_TYPE:
                    self._max_res[k] = max(self._max_res[k], node.cap[k])

        if ll_pretrained_path:
            self._load_ll(ll_pretrained_path)

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        try:
            np.save(os.path.join(directory, _LL_WEIGHTS_FILE),
                    np.array(self.ll_agent.policy_net.get_weights(), dtype=object),
                    allow_pickle=True)
            np.save(os.path.join(directory, _HL_WEIGHTS_FILE),
                    np.array(self.hl_agent.policy_net.get_weights(), dtype=object),
                    allow_pickle=True)
            self.vgae_net.save_weights(os.path.join(directory, _VGAE_WEIGHTS_FILE))
            print(f"[HRL] Models saved → {directory}")
        except Exception as e:
            print(f"[HRL] Save warning: {e}")

    def load_model(self, directory: str):
        self._load_ll(os.path.join(directory, _LL_WEIGHTS_FILE))
        self._load_hl(os.path.join(directory, _HL_WEIGHTS_FILE))
        vgae_path = os.path.join(directory, _VGAE_WEIGHTS_FILE)
        if os.path.exists(vgae_path):
            self.vgae_net.load_weights(vgae_path)

    def _load_ll(self, path: str):
        """
        Load LL weights từ file .npy.
        Nếu được truyền vào đường dẫn thư mục → tự ghép tên file chuẩn.
        Nếu được truyền đường dẫn .weights.h5 (cũ) → đổi sang .npy.
        """
        if not path:
            return

        path = self._resolve_npy_path(path, _LL_WEIGHTS_FILE)
        if not os.path.exists(path):
            print(f"[HRL] LL weights not found: {path}")
            return
        try:
            dummy = np.zeros((1, LATENT_DIM + 3), dtype=np.float32)
            self.ll_agent.policy_net(dummy)
            weights = np.load(path, allow_pickle=True)
            self.ll_agent.policy_net.set_weights(list(weights))
            print(f"[HRL] LL loaded ← {path}")
        except Exception as e:
            print(f"[HRL] LL load warning: {e}")

    def _load_hl(self, path: str):
        """Load HL weights từ file .npy."""
        if not path:
            return

        path = self._resolve_npy_path(path, _HL_WEIGHTS_FILE)
        if not os.path.exists(path):
            return
        try:
            dummy = np.zeros((1, LATENT_DIM + HighLevelAgent.FEAT_PER_SFC), dtype=np.float32)
            self.hl_agent.policy_net(dummy)
            weights = np.load(path, allow_pickle=True)
            self.hl_agent.policy_net.set_weights(list(weights))
            print(f"[HRL] HL loaded ← {path}")
        except Exception as e:
            print(f"[HRL] HL load warning: {e}")

    @staticmethod
    def _resolve_npy_path(path: str, default_filename: str) -> str:
        """
        Chuẩn hoá path về .npy:
          - Nếu là thư mục  → ghép default_filename
          - Nếu là .weights.h5 → đổi sang .npy
          - Nếu không có extension → thêm .npy
        """
        if os.path.isdir(path):
            return os.path.join(path, default_filename)
        if path.endswith(".weights.h5"):
            return path[: -len(".weights.h5")] + ".npy"
        if not path.endswith(".npy"):
            return path + ".npy"
        return path

    def get_routing(self, u: str, v: str,
                    t_start: int, t_end: int, bw: float) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]
        G = self._bw_pruned_graph(t_start, t_end, bw)
        if u not in G or v not in G:
            return None
        try:
            return nx.shortest_path(G, u, v, weight="delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _bw_pruned_graph(self, t_start: int, t_end: int, bw: float) -> nx.Graph:
        key = (t_start, t_end, round(bw, 1))
        if key in self._nx_graph_cache:
            return self._nx_graph_cache[key]
        G = nx.Graph()
        for nid in self.env.network.nodes:
            G.add_node(nid)
        for link in self.env.network.links:
            if link.get_available_bandwidth(t_start, t_end) >= bw:
                G.add_edge(link.u.name, link.v.name, delay=link.delay)
        self._nx_graph_cache[key] = G
        return G

    def _get_path_lengths(self, t_start: int, t_end: int, bw: float) -> dict:
        key = (t_start, t_end, round(bw, 1))
        if key in self._path_cache:
            return self._path_cache[key]
        G = self._bw_pruned_graph(t_start, t_end, bw)
        try:
            lengths = dict(nx.shortest_path_length(G, weight="delay"))
        except Exception:
            lengths = {}
        self._path_cache[key] = lengths
        return lengths

    def _build_dc_graph(self, t_start: int, t_end: int,
                         bw_req: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dcs = self._dc_list
        n   = len(dcs)
        if n == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 0), np.float32), []

        all_paths = self._get_path_lengths(t_start, t_end, bw_req)

        X = np.zeros((n, 3), np.float32)
        A = np.zeros((n, n), np.float32)
        for i, dc_id in enumerate(dcs):
            res  = self.env.network.nodes[dc_id].get_min_available_resource(t_start, t_end)
            X[i] = [res[k] / self._max_res[k] for k in config.RESOURCE_TYPE]
            for j, dc_j in enumerate(dcs):
                if i == j:
                    A[i, j] = 1.0
                else:
                    dist = all_paths.get(dc_id, {}).get(dc_j)
                    if dist is not None and dist > 0:
                        A[i, j] = 1.0 / (dist + 1.0)
        return X, A, dcs

    def _get_z(self, t_start: int, t_end: int,
                bw_req: float) -> Tuple[np.ndarray, List[str]]:
        key = (t_start, round(bw_req, 1))
        if key not in self._graph_cache:
            X, A, dcs = self._build_dc_graph(t_start, t_end, bw_req)
            Z = self.vgae_net.encode(X, A)
            self._graph_cache[key] = (Z, dcs, X, A)
        Z, dcs, X, A = self._graph_cache[key]
        return Z, dcs

    @staticmethod
    def _snapshot(network) -> dict:
        return {
            "nodes": {nid: {t: dict(v) for t, v in n.used.items()}
                      for nid, n in network.nodes.items()},
            "links": [{t: bw for t, bw in l.used.items()} for l in network.links],
        }

    @staticmethod
    def _restore(network, snap: dict):
        for nid, used in snap["nodes"].items():
            network.nodes[nid].used = used
        for link, used in zip(network.links, snap["links"]):
            link.used = used

    def get_placement(self, sfc: SFC, current_time: float,
                      Z_t: Optional[np.ndarray] = None,
                      dc_mapping: Optional[List[str]] = None,
                      epsilon: float = 0.0) -> Optional[Dict]:
        self._ll_traj = []
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

        if Z_t is None or dc_mapping is None:
            Z_t, dc_mapping = self._get_z(t_start, t_end, sfc.request.bw)

        if len(dc_mapping) == 0:
            return None

        node_placements, vnf_timeslots = [], []
        link_paths,      link_timeslots = [], []
        prev_dc = sfc.request.start_node

        for vnf in sfc.request.vnfs:
            cand = [str(x) for x in vnf.get_dcs()]
            if '-1' in cand or not cand:
                cand = dc_mapping
            else:
                cand = [d for d in cand if d in dc_mapping]

            valid_indices = [
                idx for idx, dc_id in enumerate(dc_mapping)
                if dc_id in cand
                   and idx < MAX_DCS
                   and self.env._check_can_deploy_vnf(
                       self.env.network.nodes[dc_id], vnf, t_start, t_end)
            ]

            if not valid_indices:
                return None

            vnf_feat   = [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
            action_idx = self.ll_agent.act(Z_t, vnf_feat, valid_indices, epsilon)
            chosen_dc  = dc_mapping[action_idx]

            path = self.get_routing(prev_dc, chosen_dc, t_start, t_end, sfc.request.bw)
            if path is None:
                return None

            self._ll_traj.append({
                "Z_t":        Z_t,
                "vnf_feat":   vnf_feat,
                "action_idx": action_idx,
                "valid_mask": valid_indices,
            })

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

    def _compute_ll_reward(self, success: bool, env_rewards: list,
                            sfc: SFC, current_time: float,
                            Z_t: np.ndarray, vnf_feat: list) -> float:
        if not success:
            return -PENALTY_DROP

        alpha, beta = self.ll_agent.get_reward_weights(Z_t, vnf_feat)
        time_rem  = max(0.0, sfc.request.end_time - current_time)
        tMax      = max(sfc.request.delay_max, 1e-6)
        raw_cost  = abs(env_rewards[1]) if len(env_rewards) > 1 else 0.0
        max_cost  = self._estimate_max_cost(sfc)
        cost_norm = min(1.0, raw_cost / max_cost)

        return R_BASE_LL + alpha * (time_rem / tMax) - beta * cost_norm

    def _greedy_placement(self, sfc: SFC, current_time: float) -> Optional[Dict]:
        from strategy.best_fit import BestFit
        return BestFit(self.env).get_placement(sfc, current_time)

    def _greedy_placement_with_traj(self, sfc, current_time, Z_t, dc_mapping):
        self._ll_traj = []
        t_start = self.env._get_timeslot(current_time)
        t_end   = self.env._get_timeslot(sfc.request.end_time)

        from strategy.best_fit import BestFit
        greedy = BestFit(self.env)
        plan   = greedy.get_placement(sfc, current_time)
        if plan is None:
            return None

        for i, vnf in enumerate(sfc.request.vnfs):
            node_plan = plan.get('nodes', {}).get(str(i))
            if node_plan is None:
                continue
            chosen_dc = node_plan['dc']
            if chosen_dc not in dc_mapping:
                continue
            action_idx = dc_mapping.index(chosen_dc)
            cand = [str(x) for x in vnf.get_dcs()]
            if '-1' in cand or not cand:
                cand = dc_mapping
            valid_indices = [
                idx for idx, dc_id in enumerate(dc_mapping)
                if dc_id in cand and idx < MAX_DCS
                and self.env._check_can_deploy_vnf(
                    self.env.network.nodes[dc_id], vnf, t_start, t_end)
            ]
            self._ll_traj.append({
                "Z_t": Z_t,
                "vnf_feat": [vnf.resource.get(k, 0.) for k in config.RESOURCE_TYPE],
                "action_idx": action_idx,
                "valid_mask": valid_indices,
            })
        return plan

    def _estimate_max_cost(self, sfc: SFC) -> float:
        key = tuple(v.name for v in sfc.request.vnfs)
        if key in self._max_cost_cache:
            return self._max_cost_cache[key]
        max_cost = 0.0
        for vnf in sfc.request.vnfs:
            node_costs = [
                n.get_cost(vnf) for n in self.env.network.nodes.values()
                if n.type == config.NODE_DC and n.cost is not None
            ]
            if node_costs:
                max_cost += max(c for c in node_costs if c < float('inf'))
        result = max(1.0, max_cost)
        self._max_cost_cache[key] = result
        return result

    def train(self) -> dict:
        total_steps         = 0
        best_acc_rate       = 0.0
        t0                  = time.time()
        total_steps_planned = self.episodes * len(self.env.requests)

        for episode in range(1, self.episodes + 1):
            ep_t0 = time.time()
            self.env.reset()

            self._graph_cache.clear()
            self._nx_graph_cache.clear()
            self._path_cache.clear()

            sfcs = sorted([SFC(r) for r in self.env.requests],
                          key=lambda s: s.request.arrival_time)
            ep_accepted = ep_rejected = 0

            for sfc in sfcs:
                total_steps += 1
                progress     = total_steps / max(1, total_steps_planned)
                epsilon      = max(0.05, 0.9 * (1.0 - progress * 3.0))
                greedy_prob  = max(0.0, 1.0 - progress * 2.0)

                self.env.t = sfc.request.arrival_time
                t_start    = self.env._get_timeslot(self.env.t)
                t_end      = self.env._get_timeslot(sfc.request.end_time)

                Z_t, dc_mapping = self._get_z(t_start, t_end, sfc.request.bw)
                cached = self._graph_cache.get((t_start, round(sfc.request.bw, 1)))
                X, A   = (cached[2], cached[3]) if cached else (None, None)

                snap       = self._snapshot(self.env.network)
                use_greedy = (np.random.random() < greedy_prob)

                if use_greedy:
                    plan          = self._greedy_placement_with_traj(sfc, self.env.t, Z_t, dc_mapping)
                    R_LL_override = 1.5
                else:
                    plan = self.get_placement(sfc, self.env.t, Z_t, dc_mapping, epsilon)
                    if plan is None:
                        plan          = self._greedy_placement_with_traj(sfc, self.env.t, Z_t, dc_mapping)
                        R_LL_override = 1.0
                    else:
                        R_LL_override = None

                success, rewards, _ = self.env.step(plan)

                if success:
                    ep_accepted += 1
                    raw_cost  = abs(rewards[1]) if len(rewards) > 1 else 0.0
                    max_cost  = self._estimate_max_cost(sfc)
                    cost_norm = min(1.0, raw_cost / max_cost)
                    R_HL      = [BASE_AR_REWARD, -cost_norm]
                    vnf_f     = ([sfc.request.vnfs[0].resource.get(k, 0.) for k in config.RESOURCE_TYPE]
                                 if sfc.request.vnfs else [0., 0., 0.])
                    R_LL      = R_LL_override if R_LL_override is not None else \
                                self._compute_ll_reward(True, rewards, sfc, self.env.t, Z_t, vnf_f)

                    self._nx_graph_cache.clear()
                    self._path_cache.clear()
                else:
                    ep_rejected += 1
                    self._restore(self.env.network, snap)
                    R_HL = [-PENALTY_DROP, 0.]
                    R_LL = -PENALTY_DROP

                Z_mean    = Z_t.mean(axis=0, keepdims=True)
                sfc_feats = self.hl_agent.extract_sfc_features([sfc], Z_t, self.ll_agent)

                sfc_pos   = sfcs.index(sfc)
                next_sfcs = sfcs[sfc_pos + 1: sfc_pos + 4]
                sfc_feats_next = (self.hl_agent.extract_sfc_features(next_sfcs, Z_t, self.ll_agent)
                                  if next_sfcs else sfc_feats)
                is_done   = (sfc is sfcs[-1])

                self.buf_HL.push((Z_mean, sfc_feats, 0, R_HL, Z_mean, sfc_feats_next, is_done))
                for i, step in enumerate(self._ll_traj):
                    nxt = self._ll_traj[i+1]["valid_mask"] if i+1 < len(self._ll_traj) else []
                    self.buf_LL.push((step["Z_t"], list(step["vnf_feat"]),
                                      step["action_idx"], R_LL, Z_t, nxt, is_done))
                if X is not None:
                    self.buf_Graph.push((X, A))

                if total_steps % 4 == 0 and len(self.buf_LL) >= BATCH_SIZE:
                    self.ll_agent.train(self.buf_LL, BATCH_SIZE)
                if total_steps % 8 == 0 and len(self.buf_HL) >= BATCH_SIZE:
                    self.hl_agent.train(self.buf_HL, BATCH_SIZE)
                if total_steps % TARGET_SYNC == 0:
                    self.ll_agent.update_target_network()
                    self.hl_agent.update_target_network()
                if total_steps % VGAE_TRAIN_FREQ == 0 and len(self.buf_Graph) >= 4:
                    self.vgae_net.train(self.buf_Graph, epochs=1)

            total_ep      = ep_accepted + ep_rejected
            acc_rate      = ep_accepted / max(1, total_ep)
            best_acc_rate = max(best_acc_rate, acc_rate)
            ep_time       = time.time() - ep_t0
            eta_s         = (self.episodes - episode) * ep_time
            eta_str       = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                             else f"{eta_s/60:.1f}m" if eta_s > 60 else f"{eta_s:.0f}s")
            cur_eps       = max(0.05, 0.9 * (1.0 - (total_steps / max(1, total_steps_planned)) * 3.0))
            bar = "█"*int(25*episode/self.episodes) + "░"*(25-int(25*episode/self.episodes))
            print(f"\r[{bar}] {episode}/{self.episodes} acc={acc_rate:.1%} best={best_acc_rate:.1%} "
                  f"ε={cur_eps:.3f} gr={greedy_prob:.2f} ETA={eta_str}", end="", flush=True)
            if episode % 25 == 0:
                print()

        print(f"\n[HRL] Done {time.time()-t0:.1f}s  best_acc={best_acc_rate:.1%}")
        self.env.stats["accepted_requests"] = ep_accepted
        self.env.stats["rejected_requests"] = ep_rejected
        self.env.stats["acceptance_ratio"]  = acc_rate
        self.env.stats["algorithm_name"]    = self.name
        return self.env.stats

    def run_simulation_eval(self) -> dict:
        self.env.reset()
        self._graph_cache.clear()
        self._nx_graph_cache.clear()
        self._path_cache.clear()

        pending = sorted([SFC(r) for r in self.env.requests],
                         key=lambda s: s.request.arrival_time)
        queue: List[SFC] = []
        accepted = rejected = 0
        t = 0.0

        while pending or queue:
            if not queue:
                if not pending:
                    break
                t = pending[0].request.arrival_time

            while pending and pending[0].request.arrival_time <= t:
                queue.append(pending.pop(0))

            expired = [s for s in queue if t > s.request.end_time]
            for s in expired:
                rejected += 1
                self.env.stats["rejected_requests"] += 1
            queue = [s for s in queue if t <= s.request.end_time]

            if not queue:
                if pending:
                    t = pending[0].request.arrival_time
                continue

            bw_req      = max(s.request.bw for s in queue)
            t_start     = self.env._get_timeslot(t)
            t_end_rough = t_start + max(
                int(max(s.request.delay_max for s in queue) / config.TIMESTEP), 10)
            Z_t, dc_mapping = self._get_z(t_start, t_end_rough, bw_req)

            sfc_idx      = self.hl_agent.act(Z_t, queue, 0.0, self.ll_agent)
            selected_sfc = queue.pop(sfc_idx)

            t_end   = self.env._get_timeslot(selected_sfc.request.end_time)
            Z_t, dc_mapping = self._get_z(t_start, t_end, selected_sfc.request.bw)

            snap       = self._snapshot(self.env.network)
            self.env.t = t
            plan = self.get_placement(selected_sfc, t, Z_t, dc_mapping, 0.0)
            if plan is None:
                plan = self._greedy_placement(selected_sfc, t)

            success, rewards, _ = self.env.step(plan) if plan else (False, [-1., 0.], -1.)

            if success:
                accepted += 1
                self.env.stats["accepted_requests"] += 1
                self.env.stats["total_cost"] += abs(rewards[1] if len(rewards) > 1 else 0.)
                self._nx_graph_cache.clear()
                self._path_cache.clear()
                if not queue and pending:
                    t = pending[0].request.arrival_time
            else:
                self._restore(self.env.network, snap)
                if t < selected_sfc.request.end_time:
                    queue.append(selected_sfc)
                    next_events = (
                        [s.request.arrival_time for s in pending if s.request.arrival_time > t] +
                        [s.request.end_time for s in queue if s.request.end_time > t]
                    )
                    t = min(next_events) if next_events else selected_sfc.request.end_time
                else:
                    rejected += 1
                    self.env.stats["rejected_requests"] += 1
                    if not queue and pending:
                        t = pending[0].request.arrival_time

        total = accepted + rejected
        self.env.stats["acceptance_ratio"] = accepted / total if total > 0 else 0.
        self.env.stats["algorithm_name"]   = self.name
        return self.env.stats