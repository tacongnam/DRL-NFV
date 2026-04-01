"""
Main entry point for NFV VNF Placement with HRL-VGAE.

Usage:
    # Training mode (auto-generates data)
    python main.py --mode train --episodes 300

    # Evaluation mode (uses test files in data/)
    python main.py --mode eval --model models/hrl_final/

    # Run baselines
    python main.py --mode baseline
"""

import os
import sys
import json
import argparse
import config
from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from strategy.fifs import GreedyFIFS
from strategy.glb import GreedyGLB
from strategy.hrl import HRL_VGAE_Strategy

# Recommended training parameters
DEFAULT_EPISODES = 300          # 300 episodes for reasonable convergence
DEFAULT_LEARNING_RATE = 0.0005  # Standard for DRL
DEFAULT_GAMMA = 0.95            # Discount factor
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 500
VGAE_TRAIN_FREQ = 100


def load_env_from_json(filepath: str) -> Env:
    with open(filepath, 'r') as f:
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
        network.add_link(
            str(link.get("u")), str(link.get("v")),
            link.get("b_l", 1.0), link.get("d_l", 1.0))

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


TRAIN_FILE = "nsf_rural_easy_s1.json"

def get_test_files(data_dir: str = "data") -> list:
    files = []
    for f in os.listdir(data_dir):
        if f.endswith('.json') and f != TRAIN_FILE:
            files.append(os.path.join(data_dir, f))
    return sorted(files)


def run_train(episodes: int, ll_pretrained: str = None, save_dir: str = "models/hrl_final", train_file: str = None):
    print("=" * 50)
    print("TRAINING MODE")
    print("=" * 50)
    print(f"Episodes: {episodes}")
    print(f"LL Pretrained: {ll_pretrained}")
    print(f"Save to: {save_dir}")
    print(f"Train file: {train_file}")

    # Use provided train file or default
    if train_file:
        if not os.path.exists(train_file):
            print(f"Error: Train file not found: {train_file}")
            return
    else:
        use_default = "data/nsf_rural_easy_s1.json"
        if not os.path.exists(use_default):
            print(f"Error: No default train file found: {use_default}")
            return
        train_file = use_default
    
    env = load_env_from_json(train_file)
    print(f"Training on: {os.path.basename(train_file)}")
    strategy = HRL_VGAE_Strategy(
        env,
        is_training=True,
        episodes=episodes,
        use_ll_score=True,
        ll_pretrained_path=ll_pretrained,
    )
    env.set_strategy(strategy)
    stats = env.run_simulation()

    os.makedirs(save_dir, exist_ok=True)
    strategy.save_model(save_dir)
    print(f"\nTraining complete. Acceptance Ratio: {stats.get('acceptance_ratio', 0):.3f}")
    print(f"Model saved to: {save_dir}")
    return strategy


def run_eval(model_dir: str = None, test_dir: str = None, test_files: list = None):
    print("=" * 50)
    print("EVALUATION MODE")
    print("=" * 50)
    if model_dir:
        print(f"Model dir: {model_dir}")
    
    # Determine test files to use
    if test_files:
        eval_files = [f if os.path.isabs(f) else os.path.join("data", f) for f in test_files]
    elif test_dir:
        eval_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
        eval_files = sorted(eval_files)
    else:
        eval_files = get_test_files("data")
    
    if not eval_files:
        print("No test files found!")
        return
    
    print(f"Testing on {len(eval_files)} file(s): {[os.path.basename(f) for f in eval_files]}")

    all_results = []
    for test_file in eval_files:
        print(f"\n--- Testing on {os.path.basename(test_file)} ---")
        env = load_env_from_json(test_file)
        strategy = HRL_VGAE_Strategy(
            env,
            is_training=False,
            use_ll_score=True,
            episodes=1,
        )

        if model_dir:
            strategy.load_model(model_dir)

        env.set_strategy(strategy)
        stats = env.run_simulation()

        all_results.append({
            'file': os.path.basename(test_file),
            'acceptance_ratio': stats.get('acceptance_ratio', 0),
            'accepted': stats.get('accepted_requests', 0),
            'rejected': stats.get('rejected_requests', 0),
            'total_cost': stats.get('total_cost', 0),
        })
        env.print_statistics()

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"{'Test File':<30} {'Acceptance':<12} {'Accepted':<10} {'Rejected':<10}")
    print("-" * 62)
    for r in all_results:
        print(f"{r['file']:<30} {r['acceptance_ratio']:<12.3f} {r['accepted']:<10} {r['rejected']:<10}")


def run_baselines():
    print("=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)

    env = load_env_from_json("data/nsf_rural_easy_s1.json")

    print("\n[1] Greedy FIFS:")
    env.set_strategy(GreedyFIFS(env))
    env.run_simulation()
    env.print_statistics()

    env.reset()
    print("[2] Greedy GLB:")
    env.set_strategy(GreedyGLB(env))
    env.run_simulation()
    env.print_statistics()


def main():
    parser = argparse.ArgumentParser(description="NFV VNF Placement with HRL-VGAE")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "baseline"],
                        help="Mode: train, eval, baseline")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Training episodes (default: {DEFAULT_EPISODES})")
    parser.add_argument("--ll-pretrained", type=str, default=None,
                        help="Path to pre-trained LL model weights")
    parser.add_argument("--model-dir", type=str, default="models/hrl_final",
                        help="Directory to save/load trained models")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Path to training data file (auto-generated via data/generate.py)")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory containing test files for eval mode")
    parser.add_argument("--test-files", type=str, nargs="+", default=None,
                        help="Specific test files to evaluate on")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(episodes=args.episodes, ll_pretrained=args.ll_pretrained,
                  save_dir=args.model_dir, train_file=args.train_file)
    elif args.mode == "eval":
        run_eval(model_dir=args.model_dir, test_dir=args.test_dir,
                 test_files=args.test_files)
    elif args.mode == "baseline":
        run_baselines()


if __name__ == "__main__":
    main()