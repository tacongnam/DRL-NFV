from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

import config
from models.placer import PressureNode
from utils.hrl_utils import restore_network

def compute_placement_reward(env, sfc, t: float, chosen_dc_name: str,
                              vnf, env_rewards: list,
                              max_cost: float,
                              path_pressure: float = 0.0) -> float:
    node    = env.network.nodes[chosen_dc_name]
    t_start = env._get_timeslot(t)
    t_end   = env._get_timeslot(sfc.request.end_time)
    res     = node.get_min_available_resource(t_start, t_end)
    cap     = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
    node_press = PressureNode.compute(res, vnf.resource, cap)
    raw_cost   = abs(env_rewards[1]) if len(env_rewards) > 1 else 0.0
    cost_norm  = min(1.0, raw_cost / max(max_cost, 1e-6))
    time_rem   = max(0.0, sfc.request.end_time - t)
    delay_norm = 1.0 - min(1.0, time_rem / max(sfc.request.delay_max, 1e-6))
    return (config.DRL_R_BASE_LL
            + config.DRL_LL_ALPHA * (1.0 - delay_norm)
            - config.DRL_LL_BETA  * cost_norm
            - node_press
            - path_pressure)

def estimate_max_cost(env, sfc) -> float:
    return max(1.0, sum(
        max((n.get_cost(v) for n in env.network.nodes.values()
             if n.type == config.NODE_DC and n.cost is not None
             and n.get_cost(v) < float('inf')), default=0.0)
        for v in sfc.request.vnfs))

def execute_with_fallback(env, plan, sfc, t: float, snap: dict):
    if plan is not None:
        success, rewards, score = env.step(plan)
        if success:
            return success, rewards, score, plan, False
        restore_network(env.network, snap)
    return False, [-1.0, 0.0], None, None, False

def rebuild_traj_from_plan(env, plan: dict, sfc, t: float,
                            Z_t: np.ndarray, dc_mapping: List[str],
                            latent_dim: int) -> List[dict]:
    from models.placer import PressureNode
    t_start      = env._get_timeslot(t)
    t_end        = env._get_timeslot(sfc.request.end_time)
    node_plan_map = {int(k.split("_")[0]): v for k, v in plan.get("nodes", {}).items()}
    prev_loc_z   = np.zeros(latent_dim, dtype=np.float32)
    traj         = []
    for i, vnf in enumerate(sfc.request.vnfs):
        np_ = node_plan_map.get(i)
        if np_ is None or np_["dc"] not in dc_mapping:
            continue
        act_idx = dc_mapping.index(np_["dc"])
        if act_idx >= config.MAX_DCS:
            continue
        cand  = [str(x) for x in vnf.get_dcs()]
        if '-1' in cand or not cand:
            cand = dc_mapping
        valid = [idx for idx, dc_id in enumerate(dc_mapping)
                 if dc_id in cand and idx < config.MAX_DCS
                 and env._check_can_deploy_vnf(env.network.nodes[dc_id], vnf, t_start, t_end)]
        node      = env.network.nodes[np_["dc"]]
        res       = node.get_min_available_resource(t_start, t_end)
        cap       = node.cap or {k: 1.0 for k in config.RESOURCE_TYPE}
        node_press = PressureNode.compute(res, vnf.resource, cap)
        traj.append({
            "Z_t":        Z_t,
            "vnf_feat":   [vnf.resource.get(k, 0.0) for k in config.RESOURCE_TYPE],
            "loc_z":      prev_loc_z,
            "node_press": node_press,
            "action_idx": act_idx,
            "valid_mask": valid,
            "dc_name":    np_["dc"],
        })
        prev_loc_z = Z_t[act_idx].copy() if act_idx < len(Z_t) else prev_loc_z
    return traj

def push_traj_to_buffer(buf, traj: List[dict],
                         Z_next: np.ndarray, R: float,
                         is_done: bool, latent_dim: int):
    for i, step in enumerate(traj):
        nxt_mask   = traj[i + 1]["valid_mask"] if i + 1 < len(traj) else []
        loc_next   = traj[i + 1]["loc_z"]      if i + 1 < len(traj) else np.zeros(latent_dim, np.float32)
        press_next = traj[i + 1]["node_press"] if i + 1 < len(traj) else 0.0
        buf.push((
            step["Z_t"], step["vnf_feat"], step["loc_z"], step["node_press"],
            step["action_idx"], R,
            Z_next, nxt_mask, loc_next, press_next, is_done,
        ))