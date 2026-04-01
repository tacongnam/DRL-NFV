"""
Pre-training script cho VGAE và Low-Level DQN.

Usage:
    python -m models.pretrain --phase vgae
    python -m models.pretrain --phase ll
    python -m models.pretrain --phase both
"""

import os
import sys
import argparse
import numpy as np
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from models.model import VGAENetwork, LowLevelAgent, ReplayBuffer
from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env


def load_env(data_path: str) -> Env:
    with open(data_path, 'r') as f:
        data = json.load(f)

    network = Network()
    vnfs = ListOfVnfs()
    requests = ListOfRequests()

    for node_id, nd in data.get("V", {}).items():
        is_server = nd.get("server", False)
        if not is_server:
            network.add_switch_node(node_id)
        else:
            network.add_dc_node(
                name=node_id,
                delay=nd.get("d_v", 0.0),
                capacity={"mem": nd.get("h_v", 1.0), "cpu": nd.get("c_v", 1.0), "ram": nd.get("r_v", 1.0)},
                cost={"mem": nd.get("cost_h", 1.0), "cpu": nd.get("cost_c", 1.0), "ram": nd.get("cost_r", 1.0)},
            )

    for link in data.get("E", []):
        network.add_link(str(link.get("u")), str(link.get("v")), link.get("b_l", 1.0), link.get("d_l", 1.0))

    for idx, vnf_data in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(
            name=idx,
            h_f=vnf_data.get("h_f", 1.0),
            c_f=vnf_data.get("c_f", 1.0),
            r_f=vnf_data.get("r_f", 1.0),
            d_f={k: v for k, v in vnf_data.get("d_f", {}).items()},
        ))

    for idx, req in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx,
            arrival_time=req.get("T", 0),
            delay_max=req.get("d_max", 100.0),
            start_node=str(req.get("st_r", "")),
            end_node=str(req.get("d_r", "")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in req.get("F_r", [])],
            bandwidth=req.get("b_r", 1.0),
        ))

    return Env(network, vnfs, requests)


def build_dc_graph(env: Env, t_start: int, t_end: int):
    dcs = [nid for nid, n in env.network.nodes.items() if n.type == config.NODE_DC]
    n = len(dcs)
    X = np.zeros((n, 3), dtype=np.float32)
    A = np.zeros((n, n), dtype=np.float32)

    for i, dc_id in enumerate(dcs):
        res = env.network.nodes[dc_id].get_min_available_resource(t_start, t_end)
        X[i] = [res["mem"], res["cpu"], res["ram"]]

    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = A[j][i] = 1.0

    return X, A, dcs


def pretrain_vgae(env: Env, epochs: int = 100, save_path: str = "models/vgae_pretrained"):
    print("PHASE 0: Pre-training VGAE")

    vgae = VGAENetwork()
    buffer = ReplayBuffer(capacity=1000)

    for _ in range(100):
        env.reset()
        t_start = np.random.randint(0, 50)
        t_end = t_start + np.random.randint(5, 20)
        X, A, _ = build_dc_graph(env, t_start, t_end)
        X_noisy = X + np.random.normal(0, 0.05, X.shape)
        buffer.push((X_noisy, A))

    print(f"Collected {len(buffer)} graph snapshots")

    for epoch in range(epochs):
        vgae.train(buffer, epochs=1)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")

    os.makedirs(save_path, exist_ok=True)
    weight_path = os.path.join(save_path, "vgae_weights.weights.h5")
    vgae.model.save_weights(weight_path)
    print(f"Saved VGAE weights to: {weight_path}")
    return vgae


def pretrain_ll(env: Env, vgae: VGAENetwork = None,
                episodes: int = 200, save_path: str = "models/ll_pretrained"):
    print("PHASE 1: Pre-training Low-Level DQN")

    latent_dim = 8
    ll_input_dim = latent_dim + 3
    ll_agent = LowLevelAgent(gamma=0.95, input_dim=ll_input_dim)
    buffer = ReplayBuffer(capacity=20000)

    if vgae is None:
        vgae = VGAENetwork()
        vgae_path = "models/vgae_pretrained/vgae_weights.weights.h5"
        if os.path.exists(vgae_path):
            vgae.model.load_weights(vgae_path)

    print(f"Training Low-Level DQN for {episodes} episodes...")
    total_rewards = []

    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.7))

        if not env.requests:
            continue

        idx = min(np.random.randint(0, len(env.requests)), len(env.requests) - 1)
        sfc = env.requests[idx]

        if not sfc.vnfs:
            continue

        t_req_start = int(sfc.arrival_time / config.TIMESTEP)
        t_req_end = t_req_start + 10

        X, A, dc_mapping = build_dc_graph(env, t_req_start, t_req_end)
        Z_t = vgae.encode(X, A)
        n_dcs = Z_t.shape[0]

        for vnf in sfc.vnfs[:5]:
            vnf_feat = np.array([[
                vnf.resource.get('mem', 0.1),
                vnf.resource.get('cpu', 0.1),
                vnf.resource.get('ram', 0.1)
            ]], dtype=np.float32)

            if np.random.random() < epsilon:
                action = np.random.randint(0, n_dcs)
            else:
                inputs = np.concatenate([Z_t, np.tile(vnf_feat, [n_dcs, 1])], axis=1)
                q_values = ll_agent.policy_net(inputs).numpy()
                action = int(np.argmax(q_values))

            target_dc_id = dc_mapping[action]
            target_dc = env.network.nodes[target_dc_id]
            can_deploy = env._check_can_deploy_vnf(target_dc, vnf, t_req_start, t_req_end)
            reward = 1.0 if can_deploy else -1.0

            buffer.push((Z_t, vnf_feat, action, reward, Z_t, list(range(n_dcs)), False))
            episode_reward += reward

        total_rewards.append(episode_reward)

        if len(buffer) > 64:
            ll_agent.train(buffer, batch_size=32)

        if (episode + 1) % 20 == 0:
            ll_agent.update_target_network()
            avg_reward = np.mean(total_rewards[-20:])
            print(f"  Ep {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Eps: {epsilon:.2f}")

    os.makedirs(save_path, exist_ok=True)
    ll_weight_path = os.path.join(save_path, "ll_dqn_weights.weights.h5")
    ll_agent.policy_net.save_weights(ll_weight_path)
    print(f"Saved LL-DQN weights to: {ll_weight_path}")
    return ll_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="both", choices=["vgae", "ll", "both"])
    parser.add_argument("--data", type=str, default="data/nsf_rural_easy_s1.json")
    parser.add_argument("--vgae-epochs", type=int, default=100)
    parser.add_argument("--ll-episodes", type=int, default=200)
    args = parser.parse_args()

    env = load_env(args.data)
    vgae = None
    ll_agent = None

    if args.phase in ["vgae", "both"]:
        vgae = pretrain_vgae(env, epochs=args.vgae_epochs)

    if args.phase in ["ll", "both"]:
        ll_agent = pretrain_ll(env, vgae=vgae, episodes=args.ll_episodes)

    print("Pre-training complete!")
    print("Usage in main.py:")
    print("  HRL_VGAE_Strategy(env, use_ll_score=True, ll_pretrained_path='models/ll_pretrained/ll_dqn_weights.weights.h5')")


if __name__ == "__main__":
    main()