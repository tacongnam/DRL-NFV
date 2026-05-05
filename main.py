import os, sys, argparse, subprocess
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

for _d in ["models/vgae_pretrained", "models/ll_pretrained",
           "models/hrl_final", "data/train", "data/test"]:
    os.makedirs(os.path.join(ROOT_DIR, _d), exist_ok=True)

sys.path.insert(0, ROOT_DIR)

from strategy import GreedyFIFS, BestFit, DeadlineAwareGreedy, HRL_VGAE_Strategy
from data.load_data import load_env_from_json, get_data_files, sample_files, save_csv
from utils import _run_eval, _run_train, _run_pretrain_inline, _plot_baseline_results, _plot_eval_vs_baselines

TRAIN_DIR        = os.path.join(ROOT_DIR, "data/train")
TEST_DIR         = os.path.join(ROOT_DIR, "data/test")
GENERATE_SCRIPT  = os.path.join(ROOT_DIR, "data/generate.py")
DEFAULT_EPISODES = 60
DEFAULT_TRAIN_REQUEST_PCT = 100
DEFAULT_PRETRAIN_REQUEST_PCT = 100

BASELINE_REGISTRY = {
    "fifs": ("GreedyFIFS", GreedyFIFS),
    "bestfit": ("BestFit", BestFit),
    "deadline": ("DeadlineAwareGreedy", DeadlineAwareGreedy),
}

def _add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train-dir", default=TRAIN_DIR)
    parser.add_argument("--model-dir", default="models/hrl_final")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-files", nargs="+", default=None)
    parser.add_argument("--ll-pretrained", type=str, default=None)
    parser.add_argument("--sample-files", type=int, default=None,
                        help="Randomly sample N files from test/baseline dir")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="Random seed for file sampling")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Path to save CSV results")

def _add_data_generation_args(parser: argparse.ArgumentParser):
    for name, default, choices in [
        ("--topology", "nsf", ["nsf", "conus", "cogent"]),
        ("--distribution", "rural", ["uniform", "rural", "urban", "centers"]),
        ("--difficulty", "easy", ["easy", "normal", "hard"]),
    ]:
        parser.add_argument(name, default=default, choices=choices)
    parser.add_argument("--scale", type=int, default=50)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--num-train-files", type=int, default=5)
    parser.add_argument("--num-test-files", type=int, default=3)

def _add_training_budget_args(parser: argparse.ArgumentParser):
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--vgae-epochs", type=int, default=60)
    parser.add_argument("--ll-episodes", type=int, default=60)
    parser.add_argument("--train-request-pct", type=int, default=DEFAULT_TRAIN_REQUEST_PCT)
    parser.add_argument("--pretrain-request-pct", type=int, default=DEFAULT_PRETRAIN_REQUEST_PCT)

def _generate_data(topology, distribution, difficulty, scale, requests,
                   num_files, output_dir, seed_offset=0):
    cmd = [
        sys.executable, "-u", GENERATE_SCRIPT,
        "--topology",     topology,
        "--distribution", distribution,
        "--difficulty",   difficulty,
        "--scale",        str(scale),
        "--num-files",    str(num_files),
        "--requests",     str(requests),
        "--seed-offset",  str(seed_offset),
        "--output",       output_dir,
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    if result.returncode != 0:
        print(f"[WARN] Command failed: {' '.join(cmd)}", flush=True)        
    return result.returncode == 0

def run_pipeline(args):
    print("\n" + "="*60)
    print("HRL-VGAE PIPELINE: generate → pretrain → train → eval")
    print("="*60)

    print("\n[1/4] Generating data ...")
    for diff, offset in [("easy", 0), ("hard", args.num_train_files)]:
        _generate_data(args.topology, args.distribution, diff,
                       args.scale, args.requests, args.num_train_files,
                       TRAIN_DIR, seed_offset=offset)
    _generate_data(args.topology, args.distribution, args.difficulty,
                   args.scale, args.requests, args.num_test_files, TEST_DIR)
    print(f"  train={len(get_data_files(TRAIN_DIR))}  test={len(get_data_files(TEST_DIR))}")

    print("\n[2/4] Pre-training VGAE + LL-DQN ...")
    ok = _run_pretrain_inline(args, TRAIN_DIR, DEFAULT_PRETRAIN_REQUEST_PCT)
    print("[2/4] Pre-training complete." if ok else "[2/4] Pre-training failed.", flush=True)

    print("\n[3/4] Training HRL ...")
    ll_path = os.path.join(ROOT_DIR, "models/ll_pretrained/ll_dqn_weights.npy")
    _run_train(
        args.episodes,
        ll_path if os.path.exists(ll_path) else None,
        os.path.join(ROOT_DIR, "models/hrl_final"),
        TRAIN_DIR,
        train_request_pct=getattr(args, "train_request_pct", DEFAULT_TRAIN_REQUEST_PCT),
    )

    print("\n[4/4] Evaluating ...")
    _run_eval(os.path.join(ROOT_DIR, "models/hrl_final"), TEST_DIR)

    print("\n" + "="*60 + "\nPIPELINE COMPLETE\n" + "="*60)

def run_generate(args):
    print("\n=== DATA GENERATION ===")
    out_train = getattr(args, "train_dir", TRAIN_DIR)
    out_test  = getattr(args, "test_dir",  None) or TEST_DIR
    for dest, n_files in [(out_train, args.num_train_files), (out_test, args.num_test_files)]:
        if n_files <= 0:
            continue
        os.makedirs(dest, exist_ok=True)
        _generate_data(args.topology, args.distribution, args.difficulty,
                       args.scale, args.requests, n_files, dest)
    print(f"train={len(get_data_files(out_train))}  test={len(get_data_files(out_test))}")

def run_pretrain(args):
    train_dir = os.path.abspath(getattr(args, "train_dir", TRAIN_DIR))
    if not get_data_files(train_dir):
        print(f"[ERROR] No JSON files in {train_dir}. Run --mode generate first.")
        return
    ok = _run_pretrain_inline(args, train_dir, DEFAULT_PRETRAIN_REQUEST_PCT)
    print("[Pretrain] Complete." if ok else "[Pretrain] Failed.", flush=True)

def run_train(args):
    print("\n=== TRAINING ===")
    ll_path = getattr(args, "ll_pretrained", None)
    if not ll_path:
        candidate = os.path.join(ROOT_DIR, "models/ll_pretrained/ll_dqn_weights.npy")
        ll_path   = candidate if os.path.exists(candidate) else None
    _run_train(args.episodes, ll_path,
               os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
               os.path.abspath(getattr(args, "train_dir", TRAIN_DIR)),
               train_request_pct=getattr(args, "train_request_pct", DEFAULT_TRAIN_REQUEST_PCT))

def run_eval(args):
    print("\n=== EVALUATION ===")
    _run_eval(
        os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
        os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR),
        getattr(args, "test_files", None),
        sample_n=getattr(args, "sample_files", None),
        sample_seed=getattr(args, "sample_seed", None),
        csv_out=getattr(args, "csv_out", None),
    )

def run_baselines(args=None):
    baselines_to_run = getattr(args, "baselines", None) or list(BASELINE_REGISTRY.keys())
    plot_out         = getattr(args, "plot_out", None)
    test_dir         = os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR)
    sample_n         = getattr(args, "sample_files", None)
    sample_seed      = getattr(args, "sample_seed", None)
    csv_out          = getattr(args, "csv_out", None)

    all_files = get_data_files(test_dir) or get_data_files(os.path.join(ROOT_DIR, "data"))
    if not all_files:
        print("[ERROR] No test files found. Run --mode generate first.")
        return

    files = sample_files(all_files, sample_n, sample_seed)
    if sample_n and len(all_files) > len(files):
        print(f"[Baseline] Sampled {len(files)}/{len(all_files)} files (seed={sample_seed})")

    print(f"\nBaseline comparison on {len(files)} file(s): {[os.path.basename(f) for f in files]}")

    csv_rows = []
    agg: dict = {}

    for fp in files:
        print(f"\n=== File: {os.path.basename(fp)} ===")
        for key in baselines_to_run:
            if key not in BASELINE_REGISTRY:
                print(f"  [WARN] Unknown baseline '{key}', skipping.")
                continue
            label, cls = BASELINE_REGISTRY[key]
            print(f"\n[{label}]")
            env = load_env_from_json(fp)
            env.set_strategy(cls(env))
            env.run_simulation()
            env.print_statistics()
            s = env.stats
            row = {
                "algorithm":        label,
                "file":             os.path.basename(fp),
                "acceptance_ratio": round(s.get("acceptance_ratio", 0.0), 4),
                "accepted":         s.get("accepted_requests", 0),
                "rejected":         s.get("rejected_requests", 0),
                "total_cost":       round(s.get("total_cost", 0.0), 2),
                "avg_cost":         round(s.get("total_cost", 0.0) / max(s.get("accepted_requests", 1), 1), 2),
                "total_delay":      round(s.get("total_delay", 0.0), 2),
                "workload served":  round(s.get("total_workload_served", 0.0), 2)
            }
            csv_rows.append(row)
            if label not in agg:
                agg[label] = {"ar": [], "cost": [], "delay": []}
            agg[label]["ar"].append(row["acceptance_ratio"])
            agg[label]["cost"].append(row["total_cost"])
            agg[label]["delay"].append(row["total_delay"])

    plot_results = []
    for label, vals in agg.items():
        plot_results.append({
            "name":  label,
            "ar":    float(np.mean(vals["ar"])),
            "cost":  float(np.mean(vals["cost"])),
            "delay": float(np.mean(vals["delay"])),
        })

    if len(plot_results) > 1:
        print("\n=== BASELINE SUMMARY (avg across files) ===")
        print(f"{'Algorithm':<25} {'AccRatio':>9} {'Cost':>10} {'Delay':>10}")
        print("-"*60)
        for r in plot_results:
            print(f"{r['name']:<25} {r['ar']:>9.3f} {r['cost']:>10.1f} {r['delay']:>10.1f}")
        _plot_baseline_results(plot_results, out_path=plot_out)

    out = csv_out or os.path.join(ROOT_DIR, "baseline_results.csv")
    save_csv(csv_rows, out, fieldnames=["algorithm","file","acceptance_ratio","accepted","rejected","total_cost","avg_cost","total_delay"])

    model_dir = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))
    hrl_weights = ["hl_pmdrl_weights.npy", "ll_dqn_weights.npy", "vgae_weights.npy"]
    if os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, w)) for w in hrl_weights
    ):
        hrl_rows = []
        hrl_agg = {"ar": [], "cost": [], "delay": []}
        for fp in files:
            print(f"\n[HRL-VGAE] Evaluating {os.path.basename(fp)} from {model_dir} ...")
            env      = load_env_from_json(fp)
            strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
            strategy.load_model(model_dir)
            env.set_strategy(strategy)
            hrl_stats = strategy.run_simulation_eval()
            env.print_statistics()
            row = {
                "algorithm":        "HRL-VGAE",
                "file":             os.path.basename(fp),
                "acceptance_ratio": round(hrl_stats.get("acceptance_ratio", 0.0), 4),
                "accepted":         hrl_stats.get("accepted_requests", 0),
                "rejected":         hrl_stats.get("rejected_requests", 0),
                "total_cost":       round(hrl_stats.get("total_cost", 0.0), 2),
                "avg_cost":         round(hrl_stats.get("average_cost", 0.0), 2),
                "total_delay":      round(hrl_stats.get("total_delay", 0.0), 2),
            }
            csv_rows.append(row)
            hrl_rows.append({"name": "HRL-VGAE", "ar": row["acceptance_ratio"],
                             "cost": row["total_cost"], "delay": row["total_delay"]})
            for k in ["ar", "cost", "delay"]:
                hrl_agg[k].append(hrl_rows[-1][k])

        hrl_plot = [{"name": "HRL-VGAE",
                     "ar":   float(np.mean(hrl_agg["ar"])),
                     "cost": float(np.mean(hrl_agg["cost"])),
                     "delay":float(np.mean(hrl_agg["delay"]))}]
        cmp_out = (plot_out.replace(".png", "_vs_hrl.png") if plot_out
                   else os.path.join(ROOT_DIR, "hrl_vs_baselines.png"))
        _plot_eval_vs_baselines(hrl_plot, plot_results, out_path=cmp_out)

        # Re-save CSV with HRL rows appended
        save_csv(csv_rows, out, fieldnames=["algorithm","file","acceptance_ratio","accepted","rejected","total_cost","avg_cost","total_delay"])

def main():
    p = argparse.ArgumentParser(description="NFV VNF Placement – HRL-VGAE")
    p.add_argument("--mode", default="baseline",
                   choices=["pipeline", "generate", "pretrain", "train", "eval", "baseline"])

    _add_data_generation_args(p)
    _add_shared_args(p)
    _add_training_budget_args(p)

    p.add_argument("--baselines",       nargs="+", default=None,
                   choices=list(BASELINE_REGISTRY.keys()))
    p.add_argument("--plot-out",        type=str, default=None)

    args = p.parse_args()

    if   args.mode == "pipeline":  run_pipeline(args)
    elif args.mode == "generate":  run_generate(args)
    elif args.mode == "pretrain":  run_pretrain(args)
    elif args.mode == "train":     run_train(args)
    elif args.mode == "eval":      run_eval(args)
    elif args.mode == "baseline":  run_baselines(args)


if __name__ == "__main__":
    main()