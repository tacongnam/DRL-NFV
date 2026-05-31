from __future__ import annotations
import math, os
import numpy as np
from typing import Dict, List, Optional, Tuple

import config
from env.env import Strategy
from env.request import SFC
from models.model import ReplayBuffer, VGAENetwork
from models.placer import PlacerAgent, PressureNode
from models.pretrain import build_dc_graph
from strategy.routing_utils import RoutingMixin
from utils.hrl_utils import snapshot_network, restore_network, LRUCache
from utils.training_logger import TrainingLogger


class DRL_Strategy(RoutingMixin, Strategy):
    def __init__(self, env, is_training: bool = False, episodes: int = 300,
                 placer_pretrained_path: str = None,
                 logger: TrainingLogger = None, episode_offset: int = 0):
        Strategy.__init__(self, env)
        RoutingMixin.__init__(self)
        self.name = "DRL-NFV"
        self.is_training = is_training
        self.episodes = episodes
        self.logger = logger
        self.episode_offset = episode_offset

        self.vgae_net = VGAENetwork(latent_dim=config.LATENT_DIM)
        input_dim = config.LATENT_DIM * 2 + PlacerAgent.FEAT_DIM_EXTRA
        self.placer = PlacerAgent(latent_dim=config.LATENT_DIM,
                                  max_dcs=config.MAX_DCS, input_dim=input_dim)
        dummy = np.zeros((1, input_dim), dtype=np.float32)
        self.placer.policy_net(dummy)
        self.placer.target_net(dummy)
        self.placer.weight_net(dummy)

        self.buf_placer = ReplayBuffer(capacity=10_000)
        self.buf_graph = ReplayBuffer(capacity=1_000)
        self._placer_traj: List[dict] = []

        self._z_cache = LRUCache(max_size=config.HRL_MAX_GRAPH_CACHE)
        self._max_res = {k: 1.0 for k in config.RESOURCE_TYPE}
        for node in env.network.nodes.values():
            if node.type == config.NODE_DC and node.cap:
                for k in config.RESOURCE_TYPE:
                    self._max_res[k] = max(self._max_res[k], node.cap[k])

        self._best_fit = None
        if placer_pretrained_path:
            self._load_placer(placer_pretrained_path)

    def load_model(self, directory: str):
        self._load_placer(os.path.join(directory, config.PLACER_WEIGHTS_FILE))
        vgae_path = os.path.join(directory, config.VGAE_WEIGHTS_FILE)
        if os.path.exists(vgae_path):
            self.vgae_net.load_weights(vgae_path)

    def _load_placer(self, path: str):
        if not path:
            return
        if os.path.isdir(path):
            path = os.path.join(path, config.PLACER_WEIGHTS_FILE)
        if not path.endswith(".npy"):
            path += ".npy"
        if not os.path.exists(path):
            print(f"[DRL] Placer weights not found: {path}")
            return
        try:
            dummy = np.zeros((1, config.LATENT_DIM * 2 + PlacerAgent.FEAT_DIM_EXTRA), np.float32)
            self.placer.policy_net(dummy)
            self.placer.policy_net.set_weights(list(np.load(path, allow_pickle=True)))
            wn = path.replace(config.PLACER_WEIGHTS_FILE, config.PLACER_WEIGHT_NET_FILE)
            if os.path.exists(wn):
                self.placer.weight_net.set_weights(list(np.load(wn, allow_pickle=True)))
            print(f"[DRL] Placer loaded <- {path}")
        except Exception as e:
            print(f"[DRL] Placer load warning: {e}")

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        try:
            np.save(os.path.join(directory, config.PLACER_WEIGHTS_FILE),
                    np.array(self.placer.policy_net.get_weights(), dtype=object),
                    allow_pickle=True)
            np.save(os.path.join(directory, config.PLACER_WEIGHT_NET_FILE),
                    np.array(self.placer.weight_net.get_weights(), dtype=object),
                    allow_pickle=True)
            self.vgae_net.save_weights(os.path.join(directory, config.VGAE_WEIGHTS_FILE))
            print(f"[DRL] Models saved -> {directory}")
        except Exception as e:
            print(f"[DRL] Save warning: {e}")

    def _get_best_fit(self):
        if self._best_fit is None:
            from strategy.best_fit import BestFit
            self._best_fit = BestFit(self.env)
        return self._best_fit

    def _get_z(self, t_start: int, t_end: int, bw_req: float,
               vnf_demand: dict = None) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        key = (t_start, t_end, round(bw_req, 1))
        cached = self._z_cache.get(key)
        if cached is not None:
            return cached
        path_cache = {}
        X, A, dcs = build_dc_graph(self.env, t_start, t_end, bw_req, path_cache, vnf_demand)
        Z = self.vgae_net.encode(X, A, deterministic=not self.is_training)
        result = (Z, dcs, X, A)
        self._z_cache.set(key, result)
        return result

    def _clear_step_caches(self):
        self.clear_routing_cache()
        self._z_cache.clear()

    def _compute_epsilon(self, progress: float) -> float:
        pw = config.EPSILON_WARMUP
        if progress < pw:
            return config.EPSILON_MAX
        t = (progress - pw) / max(1.0 - pw, 1e-6)
        return config.EPSILON_MIN + 0.5 * (config.EPSILON_MAX - config.EPSILON_MIN) * (
            1.0 + math.cos(math.pi * t))

    def _estimate_max_cost(self, sfc: SFC) -> float:
        return max(1.0, sum(
            max((n.get_cost(v) for n in self.env.network.nodes.values()
                 if n.type == config.NODE_DC and n.cost is not None
                 and n.get_cost(v) < float('inf')), default=0.0)
            for v in sfc.request.vnfs))

    def _compute_reward(self, sfc: SFC, t: float, chosen_dc_name: str,
                        vnf, env_rewards: list) -> float:
        node = self.env.network.nodes[chosen_dc_name]
        t_start = self.env._get_timeslot(t)
        t_end = self.env._get_timeslot(sfc.request.end_time)
        res = node.get_min_available_resource(t_start, t_end)
        cap = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
        node_press = PressureNode.compute(res, vnf.resource, cap)
        path_press = self.avg_path_pressure(sfc, t)
        raw_cost = abs(env_rewards[1]) if len(env_rewards) > 1 else 0.0
        cost_norm = min(1.0, raw_cost / max(self._estimate_max_cost(sfc), 1e-6))
        time_rem = max(0.0, sfc.request.end_time - t)
        delay_norm = 1.0 - min(1.0, time_rem / max(sfc.request.delay_max, 1e-6))
        return (config.HRL_R_BASE_LL
                + config.HRL_LL_ALPHA * (1.0 - delay_norm)
                - config.HRL_LL_BETA * cost_norm
                - node_press
                - path_press)

    def get_placement(self, sfc: SFC, current_time: float,
                      Z_t: np.ndarray = None, dc_mapping: List[str] = None,
                      epsilon: float = 0.0) -> Optional[Dict]:
        self._placer_traj = []
        t_start = self.env._get_timeslot(current_time)
        t_end = self.env._get_timeslot(sfc.request.end_time)

        if Z_t is None or dc_mapping is None:
            Z_t, dc_mapping, _, _ = self._get_z(t_start, t_end, sfc.request.bw)
        if not dc_mapping:
            return None

        node_placements, vnf_timeslots = [], []
        link_paths, link_timeslots = [], []
        prev_dc = sfc.request.start_node
        loc_z = np.zeros(config.LATENT_DIM, dtype=np.float32)

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

            vnf_feat = [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE]
            node_press = self._mean_candidate_pressure(valid_indices, dc_mapping, vnf, t_start, t_end)
            action_idx = self.placer.act(Z_t, vnf_feat, valid_indices, epsilon, loc_z, node_press)
            chosen_dc = dc_mapping[action_idx]

            path = self.get_routing(prev_dc, chosen_dc, t_start, t_end, sfc.request.bw)
            if path is None:
                return None

            total_delay = sum(
                link.delay for link in self.env.network.links
                if (link.u.name, link.v.name) in zip(path, path[1:])
                or (link.v.name, link.u.name) in zip(path, path[1:]))
            if total_delay > sfc.request.delay_max:
                return None

            self._placer_traj.append({
                "Z_t": Z_t, "vnf_feat": vnf_feat,
                "loc_z": loc_z.copy(), "node_press": node_press,
                "action_idx": action_idx,
                "valid_mask": valid_indices, "dc_name": chosen_dc,
            })
            loc_z = Z_t[action_idx].copy() if action_idx < len(Z_t) else loc_z
            node_placements.append(chosen_dc)
            vnf_timeslots.append((t_start, t_end))
            link_paths.append(path)
            link_timeslots.append((t_start, t_end))
            prev_dc = chosen_dc

        final_path = self.get_routing(prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None
        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))
        return self.build_placement_plan(node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)

    def _mean_candidate_pressure(self, valid_indices: List[int], dc_mapping: List[str],
                                  vnf, t_start: int, t_end: int) -> float:
        pressures = []
        for idx in valid_indices:
            dc_id = dc_mapping[idx]
            node = self.env.network.nodes[dc_id]
            res = node.get_min_available_resource(t_start, t_end)
            cap = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
            pressures.append(PressureNode.compute(res, vnf.resource, cap))
        return float(sum(pressures) / len(pressures)) if pressures else 0.0

    def _execute_with_fallback(self, plan: Optional[Dict], sfc: SFC,
                                t: float, snap: dict):
        if plan is not None:
            success, rewards, score = self.env.step(plan)
            if success:
                return success, rewards, score, plan, False
            restore_network(self.env.network, snap)

        fallback = self._get_best_fit().get_placement(sfc, t)
        if fallback is not None:
            success, rewards, score = self.env.step(fallback)
            if success:
                return success, rewards, score, fallback, True
        restore_network(self.env.network, snap)
        return False, [-1.0, 0.0], None, None, False

    def _rebuild_traj_from_plan(self, plan: dict, sfc: SFC, t: float,
                                 Z_t: np.ndarray, dc_mapping: List[str]):
        t_start = self.env._get_timeslot(t)
        t_end = self.env._get_timeslot(sfc.request.end_time)
        node_plan_map = {int(k.split("_")[0]): v for k, v in plan.get("nodes", {}).items()}
        prev_loc_z = np.zeros(config.LATENT_DIM, dtype=np.float32)
        for i, vnf in enumerate(sfc.request.vnfs):
            np_ = node_plan_map.get(i)
            if np_ is None or np_["dc"] not in dc_mapping:
                continue
            act_idx = dc_mapping.index(np_["dc"])
            if act_idx >= config.MAX_DCS:
                continue
            cand = [str(x) for x in vnf.get_dcs()]
            if '-1' in cand or not cand:
                cand = dc_mapping
            valid = [idx for idx, dc_id in enumerate(dc_mapping)
                     if dc_id in cand and idx < config.MAX_DCS
                     and self.env._check_can_deploy_vnf(
                         self.env.network.nodes[dc_id], vnf, t_start, t_end)]
            node = self.env.network.nodes[np_["dc"]]
            res = node.get_min_available_resource(t_start, t_end)
            cap = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
            node_press = PressureNode.compute(res, vnf.resource, cap)
            self._placer_traj.append({
                "Z_t": Z_t, "vnf_feat": [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE],
                "loc_z": prev_loc_z, "node_press": node_press,
                "action_idx": act_idx, "valid_mask": valid, "dc_name": np_["dc"],
            })
            prev_loc_z = Z_t[act_idx].copy() if act_idx < len(Z_t) else prev_loc_z

    @staticmethod
    def _edf_sort_key(sfc: SFC) -> float:
        return sfc.request.end_time

    def train(self) -> dict:
        total_steps = 0
        total_planned = self.episodes * len(self.env.requests)
        ep_accepted = ep_rejected = 0
        acc_rate = 0.0

        for ep in range(1, self.episodes + 1):
            self.env.reset()
            self.clear_routing_cache()
            self._z_cache.clear()
            self._best_fit = None

            pending = sorted([SFC(r) for r in self.env.requests],
                             key=lambda s: s.request.arrival_time)
            ep_accepted = ep_rejected = 0

            while pending:
                t = pending[0].request.arrival_time
                batch = []
                while pending and pending[0].request.arrival_time <= t:
                    batch.append(pending.pop(0))
                batch = [s for s in batch if t <= s.request.end_time]
                if not batch:
                    continue
                batch.sort(key=self._edf_sort_key)

                for sfc in batch:
                    total_steps += 1
                    progress = total_steps / max(1, total_planned)
                    epsilon = self._compute_epsilon(progress)

                    t_start = self.env._get_timeslot(t)
                    t_end = self.env._get_timeslot(sfc.request.end_time)
                    Z_t, dc_mapping, X, A = self._get_z(
                        t_start, t_end, sfc.request.bw,
                        sfc.request.vnfs[0].resource if sfc.request.vnfs else None)

                    snap = snapshot_network(self.env.network)
                    use_greedy = np.random.random() < max(0.05, 0.3 * (1.0 - progress))

                    if use_greedy:
                        plan = self._get_best_fit().get_placement(sfc, t)
                        self._placer_traj = []
                        if plan is not None:
                            self._rebuild_traj_from_plan(plan, sfc, t, Z_t, dc_mapping)
                    else:
                        plan = self.get_placement(sfc, t, Z_t, dc_mapping, epsilon)

                    success, rewards, score, executed_plan, used_fallback = \
                        self._execute_with_fallback(plan, sfc, t, snap)

                    if success:
                        ep_accepted += 1
                        self._clear_step_caches()
                        Z_next, _, _, _ = self._get_z(t_start, t_end, sfc.request.bw)
                        raw_cost = abs(rewards[1]) if len(rewards) > 1 else 0.0
                        cost_norm = min(1.0, raw_cost / max(self._estimate_max_cost(sfc), 1e-6))
                        time_ratio = min(1.0, (t - sfc.request.arrival_time)
                                         / max(sfc.request.delay_max, 1e-6))
                        R_placer = (config.HRL_R_BASE_LL
                                    + config.HRL_LL_ALPHA * (1.0 - time_ratio)
                                    - config.HRL_LL_BETA * cost_norm)
                        if used_fallback:
                            R_placer = -config.HRL_PENALTY_DROP * 0.5
                    else:
                        ep_rejected += 1
                        restore_network(self.env.network, snap)
                        self._clear_step_caches()
                        Z_next = Z_t
                        R_placer = -config.HRL_PENALTY_DROP

                    is_done = not pending
                    for i, step in enumerate(self._placer_traj):
                        nxt_mask = (self._placer_traj[i + 1]["valid_mask"]
                                    if i + 1 < len(self._placer_traj) else [])
                        loc_next = (self._placer_traj[i + 1]["loc_z"]
                                    if i + 1 < len(self._placer_traj)
                                    else np.zeros(config.LATENT_DIM, np.float32))
                        press_next = (self._placer_traj[i + 1]["node_press"]
                                      if i + 1 < len(self._placer_traj) else 0.0)
                        self.buf_placer.push((
                            step["Z_t"], step["vnf_feat"], step["loc_z"], step["node_press"],
                            step["action_idx"], R_placer,
                            Z_next, nxt_mask, loc_next, press_next, is_done))

                    if X is not None:
                        self.buf_graph.push((X, A))

                    if total_steps % 4 == 0 and len(self.buf_placer) >= config.HRL_BATCH_SIZE:
                        self.placer.train(self.buf_placer, config.HRL_BATCH_SIZE)

                    if total_steps % config.HRL_TARGET_SYNC == 0:
                        self.placer.update_target_network()

                    if (total_steps % config.HRL_VGAE_TRAIN_FREQ == 0
                            and len(self.buf_graph) >= 4):
                        loss = self.vgae_net.train(self.buf_graph, epochs=config.HRL_VGAE_EPOCHS)
                        if self.logger and loss is not None:
                            self.logger.log_vgae_online_train(total_steps, loss)

            total_ep = ep_accepted + ep_rejected
            acc_rate = ep_accepted / max(1, total_ep)
            if self.logger:
                self.logger.log_episode(ep + self.episode_offset, acc_rate,
                                        [ep_accepted, ep_rejected])
            if ep % 10 == 0 or ep == self.episodes:
                print(f"[DRL] ep {ep}/{self.episodes}  acc={acc_rate:.3f}", flush=True)

        self.env.stats.update({
            "accepted_requests": ep_accepted,
            "rejected_requests": ep_rejected,
            "acceptance_ratio": acc_rate,
            "algorithm_name": self.name,
        })
        return self.env.stats

    def run_simulation_eval(self) -> dict:
        self.env.reset()
        self.clear_routing_cache()
        self._z_cache.clear()
        self._best_fit = None

        pending = sorted([SFC(r) for r in self.env.requests],
                         key=lambda s: s.request.arrival_time)
        accepted = rejected = 0
        total_cost = 0.0

        while pending:
            t = pending[0].request.arrival_time
            batch = []
            while pending and pending[0].request.arrival_time <= t:
                batch.append(pending.pop(0))
            batch = [s for s in batch if t <= s.request.end_time]
            if not batch:
                continue
            batch.sort(key=self._edf_sort_key)

            for sfc in batch:
                t_start = self.env._get_timeslot(t)
                t_end = self.env._get_timeslot(sfc.request.end_time)
                Z_t, dc_mapping, _, _ = self._get_z(
                    t_start, t_end, sfc.request.bw,
                    sfc.request.vnfs[0].resource if sfc.request.vnfs else None)

                snap = snapshot_network(self.env.network)
                plan = self.get_placement(sfc, t, Z_t, dc_mapping, 0.0)
                success, rewards, _, executed_plan, _ = \
                    self._execute_with_fallback(plan, sfc, t, snap)

                if success and executed_plan:
                    accepted += 1
                    total_cost += sum(
                        self.env.network.nodes[v["dc"]].get_cost(self.env.vnfs[v["vnf_name"]])
                        for v in executed_plan.get("nodes", {}).values()
                        if v["dc"] in self.env.network.nodes
                        and v.get("vnf_name") in self.env.vnfs
                        and self.env.network.nodes[v["dc"]].get_cost(
                            self.env.vnfs[v["vnf_name"]]) < float('inf'))
                    self.env.stats["total_delay"] += (
                        sfc.request.end_time - sfc.request.arrival_time)
                else:
                    rejected += 1

                self._clear_step_caches()

        total = accepted + rejected
        self.env.stats.update({
            "accepted_requests": accepted,
            "rejected_requests": rejected,
            "total_requests": total,
            "total_cost": total_cost,
            "acceptance_ratio": accepted / total if total > 0 else 0.0,
            "average_cost": total_cost / accepted if accepted > 0 else 0.0,
            "algorithm_name": self.name,
        })
        return self.env.stats