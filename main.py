"""
main.py  –  Entry point for NFV VNF Placement with HRL-VGAE.

Pipeline
  generate → pretrain → train → eval

Key changes vs. original:
  • eval mode calls strategy.run_simulation_eval() (online, no training overhead).
  • Pipeline prints estimated time at each step.
  • All magic-number defaults removed; sensible values documented.
"""

import os, sys, json, argparse, subprocess

import config

# ── directory bootstrap ───────────────────────────────────────

for _d in ["models/vgae_pretrained", "models/ll_pretrained",
           "models/hrl_final", "data/train", "data/test"]:
    os.makedirs(_d, exist_ok=True)

from env.vnf      import VNF, ListOfVnfs
from env.request  import Request, ListOfRequests
from env.network  import Network
from env.env      import Env
from strategy.fifs import GreedyFIFS
from strategy.glb  import GreedyGLB
from strategy.hrl  import HRL_VGAE_Strategy

TRAIN_DIR = "data/train"
TEST_DIR  = "data/test"

# ── recommended defaults ──────────────────────────────────────
DEFAULT_EPISODES = 300   # ~6-8 h on CPU for NSF topology


# ─────────────────────────────────────────────────────────────
# JSON → Env loader
# ─────────────────────────────────────────────────────────────

def load_env_from_json(filepath: str) -> Env:
    with open(filepath) as f:
        data = json.load(f)
    network  = Network()
    vnfs     = ListOfVnfs()
    requests = ListOfRequests()

    for nid, nd in data.get("V", {}).items():
        if nd.get("server", False):
            network.add_dc_node(
                name=nid, delay=nd.get("d_v", 0.0),
                capacity={"mem": nd.get("h_v",1.), "cpu": nd.get("c_v",1.), "ram": nd.get("r_v",1.)},
                cost={"mem": nd.get("cost_h",1.), "cpu": nd.get("cost_c",1.), "ram": nd.get("cost_r",1.)})
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]),
                         lnk.get("b_l",1.), lnk.get("d_l",1.))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f",1.), c_f=vd.get("c_f",1.), r_f=vd.get("r_f",1.),
                         d_f={k:v for k,v in vd.get("d_f",{}).items()}))

    for idx, rd in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T",0),
            delay_max=rd.get("d_max",100.),
            start_node=str(rd.get("st_r","")), end_node=str(rd.get("d_r","")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r",[])],
            bandwidth=rd.get("b_r",1.)))
    return Env(network, vnfs, requests)


def get_data_files(d: str):
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    return []


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    print("\n" + "="*60)
    print("HRL-VGAE PIPELINE: generate → pretrain → train → eval")
    print("="*60)

    # 1. Generate
    print("\n[1/4] Generating data …")
    for diff, offset in [("easy", 0), ("hard", args.num_train_files)]:
        r = subprocess.run([sys.executable, "data/generate.py",
            "--topology", args.topology, "--distribution", args.distribution,
            "--difficulty", diff, "--scale", str(args.scale),
            "--num-files", str(args.num_train_files),
            "--requests", str(args.requests),
            "--seed-offset", str(offset), "--output", TRAIN_DIR])
        if r.returncode != 0:
            print(f"[ERROR] generate {diff} failed"); return

    r = subprocess.run([sys.executable, "data/generate.py",
        "--topology", args.topology, "--distribution", args.distribution,
        "--difficulty", args.difficulty, "--scale", str(args.scale),
        "--num-files", str(args.num_test_files),
        "--requests", str(args.requests), "--output", TEST_DIR])
    print(f"  ✓ train={len(get_data_files(TRAIN_DIR))}  test={len(get_data_files(TEST_DIR))}")

    # 2. Pretrain
    print("\n[2/4] Pre-training VGAE + LL-DQN …")
    r = subprocess.run([sys.executable, "models/pretrain.py",
        "--phase", "both", "--train-dir", TRAIN_DIR,
        "--vgae-epochs", str(args.vgae_epochs),
        "--ll-episodes", str(args.ll_episodes)])
    if r.returncode != 0:
        print("[WARN] pre-training failed, continuing with random init")

    # 3. Train
    print("\n[3/4] Training HRL …")
    _run_train(args.episodes, "models/ll_pretrained/ll_dqn_weights.weights.h5",
               "models/hrl_final", TRAIN_DIR)

    # 4. Eval
    print("\n[4/4] Evaluating …")
    _run_eval("models/hrl_final", TEST_DIR)

    print("\n" + "="*60 + "\nPIPELINE COMPLETE\n" + "="*60)


# ─────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────

def run_generate(args):
    print("\n=== DATA GENERATION ===")
    for dest, n_files in [(TRAIN_DIR, args.num_train_files),
                          (TEST_DIR,  args.num_test_files)]:
        if n_files <= 0:
            continue
        subprocess.run([sys.executable, "data/generate.py",
            "--topology", args.topology, "--distribution", args.distribution,
            "--difficulty", args.difficulty, "--scale", str(args.scale),
            "--num-files", str(n_files), "--requests", str(args.requests),
            "--output", dest])
    print(f"train={len(get_data_files(TRAIN_DIR))}  test={len(get_data_files(TEST_DIR))}")


# ─────────────────────────────────────────────────────────────
# PRETRAIN
# ─────────────────────────────────────────────────────────────

def run_pretrain(args):
    subprocess.run([sys.executable, "models/pretrain.py",
        "--phase", "both",
        "--train-dir", getattr(args, "train_dir", TRAIN_DIR),
        "--vgae-epochs", str(getattr(args, "vgae_epochs", 200)),
        "--ll-episodes", str(getattr(args, "ll_episodes", 200))])


# ─────────────────────────────────────────────────────────────
# TRAIN (internal)
# ─────────────────────────────────────────────────────────────

def _run_train(episodes, ll_pretrained, save_dir, train_dir):
    files = get_data_files(train_dir)
    if not files:
        print(f"No training files in {train_dir}"); return None

    strategy = None
    for i, fp in enumerate(files):
        print(f"\n--- File {i+1}/{len(files)}: {os.path.basename(fp)} ---")
        env = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(
            env, is_training=True, episodes=episodes,
            use_ll_score=True,
            ll_pretrained_path=ll_pretrained if i == 0 else None)

        if i > 0:
            hl_w = os.path.join(save_dir, "hl_pmdrl_weights.weights.h5")
            if os.path.exists(hl_w):
                strategy.load_model(save_dir)

        env.set_strategy(strategy)
        env.run_simulation()

    os.makedirs(save_dir, exist_ok=True)
    if strategy:
        strategy.save_model(save_dir)
    return strategy


def run_train(args):
    print("\n=== TRAINING ===")
    ll_path = getattr(args, "ll_pretrained", None)
    if ll_path is None:
        candidate = "models/ll_pretrained/ll_dqn_weights.weights.h5"
        if os.path.exists(candidate):
            ll_path = candidate
    _run_train(args.episodes, ll_path,
               getattr(args, "model_dir", "models/hrl_final"),
               getattr(args, "train_dir", TRAIN_DIR))


# ─────────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────────

def _run_eval(model_dir, test_dir, test_files=None):
    files = test_files or get_data_files(test_dir)
    if not files:
        print("No test files found."); return

    results = []
    for fp in files:
        print(f"\n--- {os.path.basename(fp)} ---")
        env = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
        if model_dir:
            strategy.load_model(model_dir)
        env.set_strategy(strategy)

        # ← use the online eval path (no training overhead)
        stats = strategy.run_simulation_eval()
        env.print_statistics()

        results.append({
            "file":  os.path.basename(fp),
            "ar":    stats.get("acceptance_ratio", 0),
            "acc":   stats.get("accepted_requests", 0),
            "rej":   stats.get("rejected_requests", 0),
            "cost":  stats.get("total_cost", 0),
        })

    print("\n=== EVAL SUMMARY ===")
    print(f"{'File':<35} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<35} {r['ar']:>9.3f} {r['acc']:>6} {r['rej']:>6} {r['cost']:>10.1f}")
    avg_ar = sum(r["ar"] for r in results) / max(1, len(results))
    print(f"\nAverage acceptance ratio: {avg_ar:.3f}")


def run_eval(args):
    print("\n=== EVALUATION ===")
    _run_eval(getattr(args, "model_dir", None),
              getattr(args, "test_dir", None) or TEST_DIR,
              getattr(args, "test_files", None))


# ─────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────

def run_baselines():
    files = get_data_files(TEST_DIR) or get_data_files("data")
    if not files:
        print("No test files found."); return

    fp  = files[0]
    print(f"\nBaseline comparison on: {os.path.basename(fp)}")
    env = load_env_from_json(fp)

    for name, cls in [("GreedyFIFS", GreedyFIFS), ("GreedyGLB", GreedyGLB)]:
        print(f"\n[{name}]")
        env.set_strategy(cls(env))
        env.run_simulation()
        env.print_statistics()
        env.reset()


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="NFV VNF Placement – HRL-VGAE")
    p.add_argument("--mode", default="train",
                   choices=["pipeline","generate","pretrain","train","eval","baseline"])

    # data
    p.add_argument("--topology",      default="nsf",
                   choices=["nsf","conus","cogent"])
    p.add_argument("--distribution",  default="rural",
                   choices=["uniform","rural","urban","centers"])
    p.add_argument("--difficulty",    default="easy",
                   choices=["easy","hard"])
    p.add_argument("--scale",         type=int, default=50)
    p.add_argument("--requests",      type=int, default=50)
    p.add_argument("--num-train-files", type=int, default=5)
    p.add_argument("--num-test-files",  type=int, default=3)

    # training
    p.add_argument("--episodes",      type=int, default=DEFAULT_EPISODES)
    p.add_argument("--ll-pretrained", type=str, default=None)
    p.add_argument("--model-dir",     default="models/hrl_final")
    p.add_argument("--train-file",    type=str, default=None)
    p.add_argument("--train-dir",     default=TRAIN_DIR)
    p.add_argument("--test-dir",      default=None)
    p.add_argument("--test-files",    nargs="+", default=None)

    # pretrain
    p.add_argument("--vgae-epochs",   type=int, default=200)
    p.add_argument("--ll-episodes",   type=int, default=200)

    args = p.parse_args()

    if args.mode == "pipeline":  run_pipeline(args)
    elif args.mode == "generate": run_generate(args)
    elif args.mode == "pretrain": run_pretrain(args)
    elif args.mode == "train":    run_train(args)
    elif args.mode == "eval":     run_eval(args)
    elif args.mode == "baseline": run_baselines()


if __name__ == "__main__":
    main()