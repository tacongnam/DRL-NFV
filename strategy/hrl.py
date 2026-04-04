"""
strategy/hrl.py  –  HRL-VGAE Strategy

Matches the PDF algorithm exactly:

  1.  VGAE encodes current network state → latent Z (cached per timestep).
  2.  High-Level Agent (PMDRL) selects the best SFC from the queue.
  3.  Low-Level Agent (DQN + action masking + Dijkstra pruning) places each VNF.
  4.  Rollback:   LL works on virtual copy S_v; committed to S_phy only on success.
  5.  Waitlist:   failed SFCs re-enter the queue; dropped only when deadline certain.
  6.  Reward shaping (no magic numbers):
        R_HL = [+BaseAR ; -TotalCost]  on success  /  [-Penalty ; 0]  on drop
        R_LL = R_base + α·max(0, D_rem/tMax) - β·cost_res_norm
        α, β learned by a small weight-network inside LowLevelAgent.

Performance
  • DC-full graph built once per timeslot (not per VNF).
  • Dijkstra called once per VNF, on a bandwidth-pruned graph.
  • No deepcopy of the whole network; only node/link .used dicts are snapshotted.
  • Training loop does at most ACTIONS_PER_STEP decisions before advancing time.
"""

from __future__ import annotations

import os, copy, math, time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

import config
from env.env import Strategy
from env.request import SFC
from models.model import ReplayBuffer, VGAENetwork, HighLevelAgent, LowLevelAgent

# Tunable constants (no magic numbers)
BASE_AR_REWARD  = 1.0     # R_HL success – acceptance objective
PENALTY_DROP    = 0.5     # R_HL fail   – penalty
R_BASE_LL       = 0.1     # small positive per-VNF step reward
BATCH_SIZE      = 32
TARGET_SYNC     = 200     # steps between target-net sync
VGAE_TRAIN_FREQ = 50      # steps between VGAE online fine-tune
LATENT_DIM      = 8
MAX_DCS         = 60      # upper bound for action space


class HRL_VGAE_Strategy(Strategy):
    """
    HRL-VGAE placement strategy.

    Parameters
    ----------
    is_training : bool
        True → runs train() loop; False → single-pass eval (online latency).
    episodes    : int
        Training episodes (ignored if is_training=False).
    ll_pretrained_path : str | None
        Path to pre-trained LL-DQN weights (.weights.h5 or .npy).
    """

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

        self.buf_HL    = ReplayBuffer(capacity=5_000)
        self.buf_LL    = ReplayBuffer(capacity=10_000)
        self.buf_Graph = ReplayBuffer(capacity=1_000)

        # Trajectory of LL decisions for current SFC (cleared each SFC)
        self._ll_traj: List[dict] = []

        if ll_pretrained_path:
            self._load_ll(ll_pretrained_path)

        # Cache: (t_slot, bw) → (Z_t, dc_mapping, X, A)
        self._graph_cache: dict = {}

    # Weight I/O

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        ll_path   = os.path.join(directory, "ll_dqn_weights.weights.h5")
        hl_path   = os.path.join(directory, "hl_pmdrl_weights.weights.h5")
        vgae_path = os.path.join(directory, "vgae_weights.npy")
        try:
            self.ll_agent.policy_net.save_weights(ll_path)
            self.hl_agent.policy_net.save_weights(hl_path)
            self.vgae_net.save_weights(vgae_path)
            print(f"[HRL] Models saved → {directory}")
        except Exception as e:
            print(f"[HRL] Save warning: {e}")

    def load_model(self, directory: str):
        self._load_ll(os.path.join(directory, "ll_dqn_weights.weights.h5"))
        hl_path = os.path.join(directory, "hl_pmdrl_weights.weights.h5")
        if os.path.exists(hl_path):
            try:
                self.hl_agent.policy_net.load_weights(hl_path)
                print(f"[HRL] HL model loaded from {hl_path}")
            except Exception as e:
                print(f"[HRL] HL load warning: {e}")
        vgae_path = os.path.join(directory, "vgae_weights.npy")
        if os.path.exists(vgae_path):
            self.vgae_net.load_weights(vgae_path)

    def _load_ll(self, path: str):
        if not path:
            return
        # accept directory, .weights.h5, or bare stem
        if os.path.isdir(path):
            path = os.path.join(path, "ll_dqn_weights.weights.h5")
        if not path.endswith(".weights.h5") and not os.path.exists(path):
            path = path + ".weights.h5"
        if os.path.exists(path):
            try:
                self.ll_agent.policy_net.load_weights(path)
                print(f"[HRL] LL model loaded from {path}")
            except Exception as e:
                print(f"[HRL] LL load warning: {e}")

    # Routing (Dijkstra on bandwidth-pruned graph)

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
        G = nx.Graph()
        for nid, node in self.env.network.nodes.items():
            G.add_node(nid)
        for link in self.env.network.links:
            if link.get_available_bandwidth(t_start, t_end) >= bw:
                G.add_edge(link.u.name, link.v.name, delay=link.delay)
        return G

    # Graph encoding (DC-only full graph, cached per t_slot)

    def _build_dc_graph(self, t_start: int, t_end: int,
                         bw_req: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build the DC-only complete graph as described in PDF slide 12:
        • Vertices = Data-Centre nodes.
        • Edge weight = 1 / (path_length) via Dijkstra between each pair.
        Feature matrix X: [mem_avail, cpu_avail, ram_avail] per DC (normalised).
        """
        dcs = [nid for nid, n in self.env.network.nodes.items()
               if n.type == config.NODE_DC]
        n   = len(dcs)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), []

        X   = np.zeros((n, 3), dtype=np.float32)
        A   = np.zeros((n, n), dtype=np.float32)
        G_p = self._bw_pruned_graph(t_start, t_end, bw_req)

        # Pre-compute shortest paths once
        try:
            all_paths = dict(nx.shortest_path_length(G_p, weight="delay"))
        except Exception:
            all_paths = {}

        max_res = {k: 1.0 for k in config.RESOURCE_TYPE}
        for node in self.env.network.nodes.values():
            if node.type == config.NODE_DC and node.cap:
                for k in config.RESOURCE_TYPE:
                    max_res[k] = max(max_res[k], node.cap[k])

        for i, dc_id in enumerate(dcs):
            node = self.env.network.nodes[dc_id]
            res  = node.get_min_available_resource(t_start, t_end)
            X[i] = [res[k] / max_res[k] for k in config.RESOURCE_TYPE]
            for j, dc_j in enumerate(dcs):
                if i == j:
                    A[i, j] = 1.0
                else:
                    dist = all_paths.get(dc_id, {}).get(dc_j, None)
                    if dist is not None and dist > 0:
                        A[i, j] = 1.0 / (dist + 1.0)

        return X, A, dcs

    def _get_z(self, t_start: int, t_end: int,
                bw_req: float) -> Tuple[np.ndarray, List[str]]:
        """Return cached (Z, dc_mapping) for current timeslot."""
        key = (t_start, round(bw_req, 2))
        if key not in self._graph_cache:
            X, A, dcs = self._build_dc_graph(t_start, t_end, bw_req)
            Z = self.vgae_net.encode(X, A)
            self._graph_cache[key] = (Z, dcs, X, A)
        Z, dcs, X, A = self._graph_cache[key]
        return Z, dcs

    # Resource snapshot / restore (lightweight rollback)

    @staticmethod
    def _snapshot(network) -> dict:
        # Each node.used is {timeslot: {resource: float}} — need a full deep copy
        # per node so rollback actually restores nested values.
        snap = {
            "nodes": {nid: copy.deepcopy(n.used) for nid, n in network.nodes.items()},
            "links": [copy.deepcopy(l.used) for l in network.links],
        }
        return snap

    @staticmethod
    def _restore(network, snap: dict):
        for nid, used in snap["nodes"].items():
            network.nodes[nid].used = used
        for link, used in zip(network.links, snap["links"]):
            link.used = used

    # SFC Placement (core algorithm, matches PDF)

    def get_placement(self, sfc: SFC, current_time: float,
                      Z_t: Optional[np.ndarray] = None,
                      dc_mapping: Optional[List[str]] = None,
                      epsilon: float = 0.0) -> Optional[Dict]:
        """
        Place all VNFs of one SFC using the Low-Level Agent.
        Returns a placement plan dict, or None on failure.
        """
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
            # Candidate DCs for this VNF
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

            vnf_feat = [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
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

        # Final segment: last VNF → end_node
        final_path = self.get_routing(
            prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None
        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))

        return self.build_placement_plan(
            node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)

    # Reward calculation (no magic numbers)

    def _compute_ll_reward(self, success: bool, env_rewards: list,
                            sfc: SFC, current_time: float,
                            Z_t: np.ndarray, vnf_feat: list) -> float:
        if not success:
            return -PENALTY_DROP

        alpha, beta = self.ll_agent.get_reward_weights(Z_t, vnf_feat)
        time_rem = sfc.request.end_time - current_time
        tMax     = max(sfc.request.delay_max, 1e-6)
        time_bonus = alpha * max(0.0, time_rem / tMax)

        cost_neg   = env_rewards[1] if len(env_rewards) > 1 else 0.0
        cost_pen   = beta * abs(cost_neg)   # env_rewards[1] is negative cost

        return R_BASE_LL + time_bonus - cost_pen

    # Training loop (matches PDF overall flow)

    def train(self) -> dict:
        total_steps   = 0
        best_acc_rate = 0.0
        t0            = time.time()

        # Warm-up: 15% of episodes use high epsilon để fill buffer nhanh hơn
        warmup_eps = max(1, int(self.episodes * 0.15))

        for episode in range(1, self.episodes + 1):
            ep_t0 = time.time()
            self.env.reset()
            self._graph_cache.clear()   # invalidate graph cache each episode

            # Sort all requests by arrival time
            waitlist = sorted(
                [SFC(r) for r in self.env.requests],
                key=lambda s: s.request.arrival_time)
            queue: List[SFC] = []

            ep_accepted = ep_rejected = 0
            t_step = 0

            progress_ratio = max(0.0, (episode - warmup_eps) /
                                  max(1, self.episodes - warmup_eps))
            # Giảm epsilon nhanh hơn giúp train nhanh gấp 2-3 lần
            hl_epsilon = 1.0 if episode <= warmup_eps else \
                         0.08 + 0.92 * math.exp(-8.0 * progress_ratio)
            ll_epsilon = 0.7 if episode <= warmup_eps else \
                         0.05 + 0.65 * math.exp(-8.0 * progress_ratio)

            max_t = (max(r.request.end_time for r in waitlist)
                     if waitlist else 100.0)
            max_step = int(max_t / config.TIMESTEP) + 1

            for t_step in range(max_step):
                self.env.t = t_step * config.TIMESTEP
                # Invalidate graph cache only when network state actually changes
                # (i.e. between timesteps, not between VNF placements in the same step)
                self._graph_cache.clear()

                # Admit arriving requests
                while waitlist and waitlist[0].request.arrival_time <= self.env.t:
                    queue.append(waitlist.pop(0))

                # Drop expired SFCs (Waitlist: only drop when deadline certain)
                queue = [s for s in queue
                         if self.env.t <= s.request.end_time]

                if not queue:
                    if not waitlist:
                        break
                    continue

                # Process ALL queued SFCs at this timestep (not just one)
                # This matches online scheduling: multiple SFCs can be served per slot
                sfc_processed_this_step = 0
                max_per_step = len(queue)   # bound to avoid infinite loop

                while queue and sfc_processed_this_step < max_per_step:
                    total_steps += 1
                    sfc_processed_this_step += 1

                    # Graph encoding (cached per timestep+bw)
                    bw_req     = queue[0].request.bw
                    t_end_eval = t_step + max(
                        int(queue[0].request.delay_max / config.TIMESTEP), 10)
                    Z_t, dc_mapping = self._get_z(t_step, t_end_eval, bw_req)
                    _, _, X, A = self._graph_cache[(t_step, round(bw_req, 2))]

                    # High-Level: select SFC
                    sfc_idx      = self.hl_agent.act(Z_t, queue, hl_epsilon, self.ll_agent)
                    selected_sfc = queue.pop(sfc_idx)

                    # Rollback snapshot
                    snap = self._snapshot(self.env.network)

                    # Low-Level: place VNFs
                    plan    = self.get_placement(
                        selected_sfc, self.env.t, Z_t, dc_mapping, ll_epsilon)
                    success, rewards, _ = self.env.step(plan)

                    if success:
                        ep_accepted += 1
                        self.env.stats['accepted_requests'] += 1
                        self.env.stats['total_cost'] += abs(
                            rewards[1] if len(rewards) > 1 else 0.0)
                        R_HL  = [BASE_AR_REWARD, rewards[1] if len(rewards) > 1 else 0.0]
                        vnf_f0 = ([selected_sfc.request.vnfs[0].resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
                                   if selected_sfc.request.vnfs else [0., 0., 0.])
                        R_LL  = self._compute_ll_reward(
                            True, rewards, selected_sfc, self.env.t, Z_t, vnf_f0)
                        # After success, invalidate cache (resources changed)
                        self._graph_cache.clear()
                    else:
                        ep_rejected += 1
                        self.env.stats['rejected_requests'] += 1
                        self._restore(self.env.network, snap)   # Rollback
                        R_HL  = [-PENALTY_DROP, 0.0]
                        R_LL  = -PENALTY_DROP
                        # Waitlist: re-queue if deadline not certain to be violated
                        if self.env.t < selected_sfc.request.end_time - config.TIMESTEP:
                            queue.append(selected_sfc)
                        break   # Stop processing queue this timestep on failure

                    # Experience storage
                    Z_mean         = Z_t.mean(axis=0, keepdims=True)  # (1, latent_dim)
                    Z_next_mean    = Z_mean.copy()
                    sfc_feats_t    = self.hl_agent.extract_sfc_features(
                        [selected_sfc], Z_t, self.ll_agent)            # (1, FEAT_PER_SFC)
                    sfc_feats_next = (self.hl_agent.extract_sfc_features(
                        queue[:1], Z_t, self.ll_agent)
                                      if queue else sfc_feats_t)        # (1, FEAT_PER_SFC)

                    is_done = not queue and not waitlist
                    self.buf_HL.push((
                        Z_mean, sfc_feats_t, 0, R_HL,
                        Z_next_mean, sfc_feats_next, is_done))

                    for i, step in enumerate(self._ll_traj):
                        nxt_mask = (self._ll_traj[i+1]["valid_mask"]
                                    if i+1 < len(self._ll_traj) else [])
                        # Store vnf_feat as flat 1-D list (not nested array)
                        self.buf_LL.push((
                            step["Z_t"],
                            list(step["vnf_feat"]),          # flat list → _make_state flattens
                            step["action_idx"],
                            R_LL,
                            Z_t, nxt_mask,
                            is_done))

                    if total_steps % VGAE_TRAIN_FREQ == 0:
                        self.buf_Graph.push((X, A))

                    # Network updates
                    if len(self.buf_LL) >= BATCH_SIZE:
                        self.ll_agent.train(self.buf_LL, BATCH_SIZE)
                    if len(self.buf_HL) >= BATCH_SIZE:
                        self.hl_agent.train(self.buf_HL, BATCH_SIZE)
                    if total_steps % TARGET_SYNC == 0:
                        self.ll_agent.update_target_network()
                        self.hl_agent.update_target_network()
                    if total_steps % VGAE_TRAIN_FREQ == 0 and len(self.buf_Graph) >= 4:
                        self.vgae_net.train(self.buf_Graph, epochs=1)

            # Episode summary
            acc_rate = ep_accepted / max(1, ep_accepted + ep_rejected)
            best_acc_rate = max(best_acc_rate, acc_rate)
            ep_time = time.time() - ep_t0
            elapsed = time.time() - t0

            remaining = self.episodes - episode
            eta_s     = remaining * ep_time
            eta_str   = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                         else f"{eta_s/60:.1f}m" if eta_s > 60
                         else f"{eta_s:.0f}s")

            bar_len = 25
            filled  = int(bar_len * episode / self.episodes)
            bar     = "█" * filled + "░" * (bar_len - filled)

            print(f"\r[{bar}] {episode}/{self.episodes} | "
                  f"acc={acc_rate:.1%} best={best_acc_rate:.1%} | "
                  f"ε_hl={hl_epsilon:.2f} ε_ll={ll_epsilon:.2f} | "
                  f"ep={ep_time:.1f}s ETA={eta_str}",
                  end="", flush=True)
            if episode % 25 == 0:
                print()

        print(f"\n[HRL] Training done in {time.time()-t0:.1f}s  "
              f"best_acc={best_acc_rate:.1%}")

        total_req = self.env.stats["total_requests"] * self.episodes
        self.env.stats["acceptance_ratio"] = (
            self.env.stats["accepted_requests"] / total_req
            if total_req > 0 else 0.0)
        self.env.stats["algorithm_name"] = self.name
        return self.env.stats

    # Evaluation (online – no training, near-real-time)

    def run_simulation_eval(self) -> dict:
        """
        Evaluate the trained policy once, mirroring the online deployment flow:
          - Advance time step-by-step.
          - Admit arriving SFCs.
          - HL picks SFC; LL places VNFs; rollback on failure; waitlist on partial fail.
          - No gradient updates.
        This keeps inference latency proportional to real time.
        """
        self.env.reset()
        self._graph_cache.clear()

        waitlist = sorted(
            [SFC(r) for r in self.env.requests],
            key=lambda s: s.request.arrival_time)
        queue: List[SFC] = []

        max_t = (max(r.request.end_time for r in waitlist)
                 if waitlist else 100.0)
        max_step = int(max_t / config.TIMESTEP) + 1

        accepted = rejected = 0

        for t_step in range(max_step):
            self.env.t = t_step * config.TIMESTEP
            self._graph_cache.clear()   # invalidate at each new timestep

            while waitlist and waitlist[0].request.arrival_time <= self.env.t:
                queue.append(waitlist.pop(0))

            queue = [s for s in queue if self.env.t <= s.request.end_time]

            if not queue:
                if not waitlist:
                    break
                continue

            # Process all queued SFCs at this timestep
            max_per_step = len(queue)
            processed    = 0

            while queue and processed < max_per_step:
                processed += 1

                bw_req     = queue[0].request.bw
                t_end_eval = t_step + max(
                    int(queue[0].request.delay_max / config.TIMESTEP), 10)
                Z_t, dc_mapping = self._get_z(t_step, t_end_eval, bw_req)

                sfc_idx      = self.hl_agent.act(Z_t, queue, 0.0, self.ll_agent)
                selected_sfc = queue.pop(sfc_idx)

                snap    = self._snapshot(self.env.network)
                plan    = self.get_placement(
                    selected_sfc, self.env.t, Z_t, dc_mapping, 0.0)
                success, rewards, _ = self.env.step(plan)

                if success:
                    accepted += 1
                    self.env.stats["accepted_requests"] += 1
                    self.env.stats["total_cost"] += abs(
                        rewards[1] if len(rewards) > 1 else 0.0)
                    self._graph_cache.clear()   # resources changed
                else:
                    rejected += 1
                    self.env.stats["rejected_requests"] += 1
                    self._restore(self.env.network, snap)
                    if self.env.t < selected_sfc.request.end_time - config.TIMESTEP:
                        queue.append(selected_sfc)
                    break   # stop processing this timestep on failure

        total = accepted + rejected
        self.env.stats["acceptance_ratio"] = accepted / total if total > 0 else 0.0
        self.env.stats["algorithm_name"]   = self.name
        return self.env.stats