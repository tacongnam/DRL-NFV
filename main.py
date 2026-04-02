"""
Main entry point for NFV VNF Placement with HRL-VGAE.

Complete Pipeline:
    generate -> pre-train -> train -> eval

Usage:
    # ===== FULL AUTOMATED PIPELINE =====
    # Generate data, pre-train, train HRL, and evaluate
    python main.py --mode pipeline --topology nsf --distribution rural --difficulty easy \
                   --num-train-files 5 --num-test-files 3 --episodes 300

    # ===== INDIVIDUAL STEPS =====
    
    # Step 1: Generate training/test data
    python main.py --mode generate --topology nsf --distribution rural --difficulty easy \
                   --num-train-files 10 --num-test-files 3
    
    # Step 2: Pre-train VGAE and Low-Level DQN
    python main.py --mode pretrain --train-dir data/train
    
    # Step 3: Train HRL policy
    python main.py --mode train --episodes 300 --ll-pretrained models/ll_pretrained
    
    # Step 4: Evaluate trained model
    python main.py --mode eval --model-dir models/hrl_final --test-dir data/test

    # ===== BASELINES =====
    python main.py --mode baseline
"""

import os
import sys
import json
import argparse
import subprocess
import config


def ensure_directories():
    """Create required directories if they don't exist."""
    dirs = [
        "models/vgae_pretrained",
        "models/ll_pretrained",
        "models/hrl_final",
        "data/train",
        "data/test"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# Ensure dirs on startup
ensure_directories()

from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from strategy.fifs import GreedyFIFS
from strategy.glb import GreedyGLB
from strategy.hrl import HRL_VGAE_Strategy

# Recommended training parameters - Optimized for 6-10 hour training
DEFAULT_EPISODES = 100          # Reduced from 300 to 100 for faster training
DEFAULT_LEARNING_RATE = 0.0005  # Standard for DRL
DEFAULT_GAMMA = 0.95            # Discount factor
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 200        # Reduced from 500 for more frequent updates
VGAE_TRAIN_FREQ = 50            # Reduced from 100 for more frequent VGAE updates


def load_env_from_json(filepath: str) -> Env:
    """Load environment from JSON file."""
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


TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def get_data_files(data_dir: str) -> list:
    """Get JSON files from directory."""
    if os.path.exists(data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return sorted(files)
    return []

def run_pipeline(args):
    """Run the complete pipeline: generate -> pretrain -> train -> eval."""
    print("\n" + "=" * 60)
    print("DRL-NFV COMPLETE PIPELINE")
    print("Generate -> Pre-train -> Train -> Evaluate")
    print("=" * 60 + "\n")

    print("\n[STEP 1/4] Generating training and test data...")
    print("-" * 40)
    num_easy_train = 5
    num_hard_train = 5
    
    print(f"Generating {num_easy_train} easy training files...")
    result = subprocess.run([
        sys.executable, "data/generate.py",
        "--topology", args.topology,
        "--distribution", args.distribution,
        "--difficulty", "easy",
        "--scale", str(args.scale),
        "--num-files", str(num_easy_train),
        "--requests", str(args.requests),
        "--output", TRAIN_DIR
    ])
    if result.returncode != 0:
        print("Error: Generating easy training data failed!")
        return

    print(f"Generating {num_hard_train} hard training files...")
    result = subprocess.run([
        sys.executable, "data/generate.py",
        "--topology", args.topology,
        "--distribution", args.distribution,
        "--difficulty", "hard",
        "--scale", str(args.scale),
        "--num-files", str(num_hard_train),
        "--requests", str(args.requests),
        "--seed-offset", str(num_easy_train),
        "--output", TRAIN_DIR
    ])
    if result.returncode != 0:
        print("Error: Generating hard training data failed!")
        return

    # Generate test data
    print("\nGenerating test data...")
    generate_test_cmd = [
        sys.executable, "data/generate.py",
        "--topology", args.topology,
        "--distribution", args.distribution,
        "--difficulty", args.difficulty,
        "--scale", str(args.scale),
        "--num-files", str(args.num_test_files),
        "--requests", str(args.requests),
        "--output", TEST_DIR
    ]
    print(f"Command: {' '.join(generate_test_cmd)}")
    result = subprocess.run(generate_test_cmd)
    if result.returncode != 0:
        print("Warning: Test data generation failed, continuing anyway")

    train_files = get_data_files(TRAIN_DIR)
    test_files = get_data_files(TEST_DIR)
    print(f"\n✓ Generated {len(train_files)} training files, {len(test_files)} test files")

    # Step 2: Pre-train
    print("\n[STEP 2/4] Pre-training VGAE and Low-Level DQN...")
    print("-" * 40)
    pretrain_cmd = [
        sys.executable, "models/pretrain.py",
        "--phase", "both",
        "--train-dir", TRAIN_DIR,
        "--vgae-epochs", str(args.vgae_epochs),
        "--ll-episodes", str(args.ll_episodes)
    ]
    print(f"Command: {' '.join(pretrain_cmd)}")
    result = subprocess.run(pretrain_cmd)
    if result.returncode != 0:
        print("Warning: Pre-training failed, will continue with random initialization")

    # Step 3: Train HRL
    print("\n[STEP 3/4] Training HRL policy...")
    print("-" * 40)
    ll_pretrained = "models/ll_pretrained"
    ll_weights_path = os.path.join(ll_pretrained, "ll_dqn_weights.weights.h5")
    if os.path.exists(ll_weights_path):
        print(f"Using pre-trained LL model from: {ll_weights_path}")
        ll_pretrained = ll_weights_path  # Pass the full file path
    else:
        print("No pre-trained LL model found, training from scratch")
        ll_pretrained = None

    strategy = run_train_internal(
        episodes=args.episodes,
        ll_pretrained=ll_pretrained,
        save_dir="models/hrl_final",
        train_dir=TRAIN_DIR
    )

    # Step 4: Evaluate
    print("\n[STEP 4/4] Evaluating trained model on test data...")
    print("-" * 40)
    run_eval(
        model_dir="models/hrl_final",
        test_dir=TEST_DIR
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Training data: {TRAIN_DIR}/")
    print(f"Test data: {TEST_DIR}/")
    print(f"Pre-trained models: models/vgae_pretrained/, models/ll_pretrained/")
    print(f"Final HRL model: models/hrl_final/")


def run_train_internal(episodes: int, ll_pretrained: str = None, 
                       save_dir: str = "models/hrl_final", train_dir: str = TRAIN_DIR):
    """Internal training function used by pipeline and train mode."""
    eval_files = get_data_files(train_dir)
    
    if not eval_files:
        print(f"Error: No training files found in {train_dir}")
        return None
    
    print(f"Found {len(eval_files)} training file(s)")
    
    strategy = None
    all_results = []
    pretrained_path = ll_pretrained  # Save original pretrained path
    
    for i, t_file in enumerate(eval_files):
        print(f"\n--- Training on {os.path.basename(t_file)} ({i+1}/{len(eval_files)}) ---")
        env = load_env_from_json(t_file)
        
        strategy = HRL_VGAE_Strategy(
            env,
            is_training=True,
            episodes=episodes,
            use_ll_score=True,
            ll_pretrained_path=pretrained_path if i == 0 else None,
        )
        
        # Load previous model weights if continuing training
        if i > 0 and os.path.exists(os.path.join(save_dir, "hl_pmdrl_weights.weights.h5")):
            strategy.load_model(save_dir)
        
        env.set_strategy(strategy)
        stats = env.run_simulation()
        all_results.append({'file': os.path.basename(t_file), 'stats': stats})
    
    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    if strategy:
        strategy.save_model(save_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"{'File':<40} {'Acceptance':<12} {'Accepted':<10}")
    print("-" * 62)
    for r in all_results:
        stats = r['stats']
        acc_ratio = stats.get('acceptance_ratio', 0)
        acc_req = stats.get('accepted_requests', 0)
        print(f"{r['file']:<40} {acc_ratio:<12.3f} {acc_req:<10}")
    
    print(f"\nModel saved to: {save_dir}")
    return strategy


# ============================================================
# GENERATE MODE
# ============================================================

def run_generate(args):
    """Run data generation."""
    print("\n" + "=" * 50)
    print("DATA GENERATION")
    print("=" * 50)

    # Generate training data
    if args.num_train_files > 0:
        print(f"\nGenerating {args.num_train_files} training file(s) to {TRAIN_DIR}/...")
        generate_cmd = [
            sys.executable, "data/generate.py",
            "--topology", args.topology,
            "--distribution", args.distribution,
            "--difficulty", args.difficulty,
            "--scale", str(args.scale),
            "--num-files", str(args.num_train_files),
            "--requests", str(args.requests),
            "--output", TRAIN_DIR
        ]
        result = subprocess.run(generate_cmd)
        if result.returncode != 0:
            print("Error: Training data generation failed!")
            return

    # Generate test data
    if args.num_test_files > 0:
        print(f"\nGenerating {args.num_test_files} test file(s) to {TEST_DIR}/...")
        generate_test_cmd = [
            sys.executable, "data/generate.py",
            "--topology", args.topology,
            "--distribution", args.distribution,
            "--difficulty", args.difficulty,
            "--scale", str(args.scale),
            "--num-files", str(args.num_test_files),
            "--requests", str(args.requests),
            "--output", TEST_DIR
        ]
        result = subprocess.run(generate_test_cmd)
        if result.returncode != 0:
            print("Error: Test data generation failed!")
            return

    train_files = get_data_files(TRAIN_DIR)
    test_files = get_data_files(TEST_DIR)
    print(f"\n✓ Total: {len(train_files)} training files, {len(test_files)} test files")


# ============================================================
# PRETRAIN MODE
# ============================================================

def run_pretrain(args):
    """Run pre-training."""
    print("\n" + "=" * 50)
    print("PRE-TRAINING")
    print("=" * 50)

    pretrain_cmd = [
        sys.executable, "models/pretrain.py",
        "--phase", "both",
        "--train-dir", args.train_dir if hasattr(args, 'train_dir') and args.train_dir else TRAIN_DIR,
        "--vgae-epochs", str(args.vgae_epochs) if hasattr(args, 'vgae_epochs') else "100",
        "--ll-episodes", str(args.ll_episodes) if hasattr(args, 'll_episodes') else "200"
    ]
    result = subprocess.run(pretrain_cmd)
    return result.returncode == 0


# ============================================================
# TRAIN MODE
# ============================================================

def run_train(episodes: int, ll_pretrained: str = None, save_dir: str = "models/hrl_final",
              train_file: str = None, train_dir: str = TRAIN_DIR, train_files: list = None):
    """Train HRL policy."""
    print("=" * 50)
    print("TRAINING MODE")
    print("=" * 50)
    print(f"Episodes: {episodes}")
    print(f"LL Pretrained: {ll_pretrained}")
    print(f"Save to: {save_dir}")

    # Determine training files
    if train_files:
        eval_files = [f if os.path.isabs(f) else os.path.join("data", f) for f in train_files]
        eval_files = [f for f in eval_files if os.path.exists(f)]
    elif train_file:
        eval_files = [train_file] if os.path.exists(train_file) else []
    else:
        eval_files = get_data_files(train_dir)

    if not eval_files:
        print(f"Error: No training files found in {train_dir}")
        return

    print(f"Found {len(eval_files)} training file(s)")

    strategy = None
    all_results = []
    pretrained_path = ll_pretrained

    for i, t_file in enumerate(eval_files):
        print(f"\n--- Training on {os.path.basename(t_file)} ({i+1}/{len(eval_files)}) ---")
        env = load_env_from_json(t_file)

        strategy = HRL_VGAE_Strategy(
            env,
            is_training=True,
            episodes=episodes,
            use_ll_score=True,
            ll_pretrained_path=pretrained_path if i == 0 else None,
        )

        # Load previous model weights if continuing training
        if i > 0 and os.path.exists(os.path.join(save_dir, "hl_pmdrl_weights.weights.h5")):
            strategy.load_model(save_dir)

        env.set_strategy(strategy)
        stats = env.run_simulation()
        all_results.append({'file': os.path.basename(t_file), 'stats': stats})

    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    if strategy:
        strategy.save_model(save_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"{'File':<40} {'Acceptance':<12} {'Accepted':<10}")
    print("-" * 62)
    for r in all_results:
        stats = r['stats']
        acc_ratio = stats.get('acceptance_ratio', 0)
        acc_req = stats.get('accepted_requests', 0)
        print(f"{r['file']:<40} {acc_ratio:<12.3f} {acc_req:<10}")

    print(f"\nModel saved to: {save_dir}")
    return strategy


# ============================================================
# EVAL MODE
# ============================================================

def run_eval(model_dir: str = None, test_dir: str = None, test_files: list = None):
    """Evaluate trained model."""
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
        eval_files = get_data_files(TEST_DIR)

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


# ============================================================
# BASELINE MODE
# ============================================================

def run_baselines():
    """Run baseline strategies."""
    print("=" * 50)
    print("BASELINE COMPARISON")
    print("=" * 50)

    # Find a test file to run baselines on
    test_files = get_data_files(TEST_DIR) or get_data_files("data")
    if not test_files:
        print("No test files found for baseline comparison!")
        return

    for test_file in test_files[:1]:  # Run on first test file
        print(f"\nTesting on: {os.path.basename(test_file)}")
        env = load_env_from_json(test_file)

        print("\n[1] Greedy FIFS:")
        env.set_strategy(GreedyFIFS(env))
        env.run_simulation()
        env.print_statistics()

        env.reset()
        print("\n[2] Greedy GLB:")
        env.set_strategy(GreedyGLB(env))
        env.run_simulation()
        env.print_statistics()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NFV VNF Placement with HRL-VGAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full automated pipeline
  python main.py --mode pipeline --topology nsf --distribution rural --difficulty easy \\
                 --num-train-files 5 --num-test-files 3 --episodes 300

  # Step by step
  python main.py --mode generate --num-train-files 10 --num-test-files 3
  python main.py --mode pretrain
  python main.py --mode train --episodes 300
  python main.py --mode eval
        """
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["pipeline", "generate", "pretrain", "train", "eval", "baseline"],
                        help="Mode: pipeline, generate, pretrain, train, eval, baseline")

    # Data generation arguments
    parser.add_argument("--topology", type=str, default="nsf",
                        choices=["nsf", "conus", "cogent"],
                        help="Network topology")
    parser.add_argument("--distribution", type=str, default="rural",
                        choices=["uniform", "rural", "urban", "centers"],
                        help="Server node distribution")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "hard"],
                        help="Difficulty level")
    parser.add_argument("--scale", type=int, default=50,
                        help="Resource scale factor")
    parser.add_argument("--requests", type=int, default=50,
                        help="Number of requests per file")
    parser.add_argument("--num-train-files", type=int, default=5,
                        help="Number of training files to generate")
    parser.add_argument("--num-test-files", type=int, default=3,
                        help="Number of test files to generate")

    # Training arguments
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Training episodes (default: {DEFAULT_EPISODES})")
    parser.add_argument("--ll-pretrained", type=str, default=None,
                        help="Path to pre-trained LL model weights")
    parser.add_argument("--model-dir", type=str, default="models/hrl_final",
                        help="Directory to save/load trained models")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Path to training data file")
    parser.add_argument("--train-dir", type=str, default=TRAIN_DIR,
                        help="Directory containing training files")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Directory containing test files for eval mode")
    parser.add_argument("--test-files", type=str, nargs="+", default=None,
                        help="Specific test files to evaluate on")

    # Pre-training arguments
    parser.add_argument("--vgae-epochs", type=int, default=100,
                        help="VGAE pre-training epochs")
    parser.add_argument("--ll-episodes", type=int, default=200,
                        help="LL-DQN pre-training episodes")

    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "generate":
        run_generate(args)
    elif args.mode == "pretrain":
        run_pretrain(args)
    elif args.mode == "train":
        run_train(episodes=args.episodes, ll_pretrained=args.ll_pretrained,
                  save_dir=args.model_dir, train_file=args.train_file,
                  train_dir=args.train_dir)
    elif args.mode == "eval":
        run_eval(model_dir=args.model_dir, test_dir=args.test_dir,
                 test_files=args.test_files)
    elif args.mode == "baseline":
        run_baselines()


if __name__ == "__main__":
    main()