from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import copy
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import networkx as nx

from env.network import Network, Link, Node
from env.request import Request, SFC, ListOfRequests
from env.vnf import VNF, ListOfVnfs
import config

class Strategy(ABC):
    def __init__(self, env: Env):
        self.env  = env
        self.name = "PlacementStrategy"

    @abstractmethod
    def get_routing(self, u: str, v: str, t_start: int, t_end: int, bw: float) -> List:
        pass

    @abstractmethod
    def get_placement(self, sfc: SFC, current_time: float) -> Optional[Dict]:
        pass

    def build_placement_plan(
        self,
        node_placements: List[str],
        link_paths:      List[List[str]],
        vnf_timeslots:   List[tuple],
        link_timeslots:  List[tuple],
        sfc:             SFC,
    ) -> Dict:
        plan = {'nodes': {}, 'links': {}, 'bw': sfc.request.bw}

        for i, (dc, (T1, T2)) in enumerate(zip(node_placements, vnf_timeslots)):
            plan['nodes'][str(i)] = {'dc': dc, 'T1': T1, 'T2': T2}

        for i, (path, (T1, T2)) in enumerate(zip(link_paths, link_timeslots)):
            plan['links'][str(i)] = {'path': path, 'T1': T1, 'T2': T2}

        return plan

class Env(gym.Env):
    """
    NFV placement environment.

    Kế thừa gym.Env để tương thích với stable-baselines3 / RLlib.
    Vòng lặp chính được điều khiển bởi Strategy.train() / run_simulation(),
    không phải vòng lặp gym chuẩn. observation_space và action_space được khai
    báo symbolic — override nếu tích hợp thư viện RL bên ngoài.
    """

    _OBS_DIM = 1
    _ACT_DIM = 1

    def __init__(self, network: Network, vnfs: ListOfVnfs, requests: ListOfRequests):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self._ACT_DIM)

        self.network  = network
        self.vnfs     = vnfs.vnfs          # dict {str(idx): VNF}
        self.requests = requests.requests
        self.waitlist: list = []
        self.strategy: Optional[Strategy] = None
        self.t        = 0.0
        self.stats    = self._empty_stats()

    def _empty_stats(self) -> dict:
        return {
            'total_requests':    len(self.requests),
            'accepted_requests': 0,
            'rejected_requests': 0,
            'total_cost':        0.0,
            'total_delay':       0.0,
            'accepted_details':  [],
            'rejected_details':  [],
        }

    def _reinit_network_usage(self):
        """Xóa usage về 0 — dùng reinit thay vì .clear() để giữ timeslot 0."""
        for n in self.network.nodes.values():
            if n.type == config.NODE_DC:
                n.used = {0: {k: 0.0 for k in config.RESOURCE_TYPE}}
            else:
                n.used = {}
        for l in self.network.links:
            l.used = {0: 0.0}

    def reset(self, *, seed=None, options=None):
        """Gymnasium >= 0.26: trả về (obs, info)."""
        super().reset(seed=seed)
        self._reinit_network_usage()
        self.waitlist = []
        self.t        = 0.0
        self.stats    = self._empty_stats()
        return np.zeros(self._OBS_DIM, dtype=np.float32), {}

    def set_strategy(self, strategy: Strategy):
        self.strategy = strategy

    def _get_timeslot(self, t: float) -> int:
        return int(round(t / config.TIMESTEP))

    def _check_can_deploy_vnf(self, node: Node, vnf: VNF,
                           t_start: int, t_end: int) -> bool:
        if node.type == config.NODE_SWITCH or node.cap is None:
            return False
        relevant = [t for t in node.used if t_start <= t <= t_end]
        if not relevant:
            used_before = max((t for t in node.used if t < t_start), default=None)
            base = node.used[used_before] if used_before is not None else {k: 0.0 for k in config.RESOURCE_TYPE}
            return not any(base[k] + vnf.resource[k] > node.cap[k] for k in config.RESOURCE_TYPE)
        return all(
            not any(node.used[t][k] + vnf.resource[k] > node.cap[k] for k in config.RESOURCE_TYPE)
            for t in relevant
        )

    def step(self, plan: Optional[Dict]) -> Tuple[bool, List[float], float]:
        """
        Thực thi placement plan.
        Returns: (success, [reward_acceptance, reward_cost], score)
        """
        if plan is None:
            return False, [-1.0, 0.0], -1.0

        backup = copy.deepcopy(self.network)
        cost   = 0.0

        try:
            # 1. Deploy VNFs lên nodes
            for vnf_name, node_plan in plan.get('nodes', {}).items():
                dc_name = node_plan['dc']
                T1, T2  = node_plan['T1'], node_plan['T2']
                dc  = self.network.nodes[dc_name]
                vnf = self.vnfs[vnf_name]

                if not self._check_can_deploy_vnf(dc, vnf, T1, T2):
                    self.network = backup
                    return False, [-1.0, 0.0], -1.0

                dc.use(vnf.resource, T1, T2 + 1)
                cost += dc.get_cost(vnf)

            # 2. Allocate bandwidth trên các path
            sfc_bw = plan.get('bw', 1.0)
            for link_plan in plan.get('links', {}).values():
                path    = link_plan['path']
                T1, T2  = link_plan['T1'], link_plan['T2']

                for u, v in zip(path[:-1], path[1:]):
                    target = next(
                        (l for l in self.network.links
                         if {l.u.name, l.v.name} == {u, v}),
                        None
                    )
                    if target is None:
                        self.network = backup
                        return False, [-1.0, 0.0], -1.0

                    # FIX: kiểm tra bandwidth trước khi commit
                    if any(target.check_violated(t, sfc_bw) for t in range(T1, T2 + 1)):
                        self.network = backup
                        return False, [-1.0, 0.0], -1.0

                    target.use(sfc_bw, T1, T2 + 1)

            reward_cost = -cost
            score       = 1.0 + 0.1 * reward_cost
            return True, [1.0, reward_cost], score

        except Exception as e:
            print(f"[Env.step] Error: {e}")
            self.network = backup
            return False, [-1.0, 0.0], -1.0

    def process_request(self, sfc: SFC) -> Tuple[bool, List[float], float]:
        if self.strategy is None:
            raise RuntimeError("No strategy set — call env.set_strategy() first")

        plan    = self.strategy.get_placement(sfc, self.t)
        success, rewards, score = self.step(plan)

        if success:
            self.stats['accepted_requests'] += 1
            self.stats['total_cost']  += -rewards[1]
            self.stats['total_delay'] += sfc.request.end_time - sfc.request.arrival_time
            self.stats['accepted_details'].append({
                'request_id':   sfc.request.name,
                'arrival_time': sfc.request.arrival_time,
                'score':        score,
            })
        else:
            self.stats['rejected_requests'] += 1
            self.stats['rejected_details'].append({
                'request_id':   sfc.request.name,
                'arrival_time': sfc.request.arrival_time,
            })

        return success, rewards, score

    def run_simulation(self) -> Dict:
        if self.strategy is None:
            raise RuntimeError("No strategy set — call env.set_strategy() first")

        # DRL strategy với is_training=True → nhường vòng lặp cho Strategy.train()
        if getattr(self.strategy, 'is_training', False) and hasattr(self.strategy, 'train'):
            return self.strategy.train()

        # Baseline: chạy tuần tự
        self.reset()   # trả về (obs, info) — không cần unpack ở đây
        for req in sorted(self.requests, key=lambda r: r.arrival_time):
            self.t = req.arrival_time
            self.process_request(SFC(req))

        accepted = self.stats['accepted_requests']
        total    = self.stats['total_requests']
        self.stats['average_cost']     = (self.stats['total_cost'] / accepted
                                          if accepted > 0 else 0.0)
        self.stats['acceptance_ratio'] = accepted / total if total > 0 else 0.0
        self.stats['algorithm_name']   = self.strategy.name
        return self.stats

    def _run_rl_simulation(self):
        """Evaluate mô hình RL đã hội tụ theo vòng lặp timeslot."""
        sorted_req = sorted(self.requests, key=lambda r: r.arrival_time)
        req_idx    = 0

        # FIX: tính max_time tự động nếu chưa được set từ bên ngoài
        max_time = getattr(self, 'max_time', None)
        if max_time is None:
            max_time = max((r.end_time for r in self.requests), default=0.0)

        for t_step in range(int(max_time / config.TIMESTEP) + 1):
            self.t = t_step * config.TIMESTEP

            while req_idx < len(sorted_req) and sorted_req[req_idx].arrival_time <= self.t:
                self.waitlist.append(SFC(sorted_req[req_idx]))
                req_idx += 1

            valid, expired = [], []
            for sfc in self.waitlist:
                (valid if self.t <= sfc.request.end_time else expired).append(sfc)
            self.stats['rejected_requests'] += len(expired)
            self.waitlist = valid

            while self.waitlist:
                if not self.strategy.process_waitlist():
                    break

    def print_statistics(self):
        s = self.stats
        print("\n" + "=" * 30)
        print("      THỐNG KÊ CHI TIẾT")
        print("=" * 30)
        print(f"{'Tổng số request:':<22} {s['total_requests']}")
        print(f"{'Đã chấp nhận:':<22} {s['accepted_requests']}")
        print(f"{'Đã từ chối:':<22} {s['rejected_requests']}")
        print("-" * 30)
        print(f"{'Tổng chi phí:':<22} {s['total_cost']:.2f}")
        print(f"{'Tổng độ trễ:':<22} {s['total_delay']:.2f}")
        if s['accepted_requests'] > 0:
            avg_d = s['total_delay'] / s['accepted_requests']
            print(f"{'Độ trễ trung bình:':<22} {avg_d:.2f}")
        if 'acceptance_ratio' in s:
            print(f"{'Acceptance ratio:':<22} {s['acceptance_ratio']:.3f}")
        print("=" * 30 + "\n")