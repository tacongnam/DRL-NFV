import os, sys, argparse, subprocess, time
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

for _d in ["models/vgae_pretrained", "models/placer",
           "models/hrl_final", "data/train", "data/test"]:
    os.makedirs(os.path.join(ROOT_DIR, _d), exist_ok=True)

sys.path.insert(0, ROOT_DIR)

import config
from strategy import GreedyFIFS, BestFit, DeadlineAwareGreedy, RandomFit, ShortestPathFirst, GreedyGLB, DRL_Strategy
from data.load_data import load_env_from_json, get_data_files, save_csv
from utils import _run_eval, _run_train, _run_pretrain_inline, _plot_baseline_results, _plot_eval_vs_baselines
from utils.training_logger import TrainingLogger

TRAIN_DIR = os.path.join(ROOT_DIR, "data/train")
TEST_DIR = os.path.join(ROOT_DIR, "data/test")
GENERATE_SCRIPT = os.path.join(ROOT_DIR, "data/generate.py")
DEFAULT_EPISODES = 60

BASELINE_REGISTRY = {
    "fifs": ("GreedyFIFS", GreedyFIFS),
    "bestfit": ("BestFit", BestFit),
    "deadline": ("DeadlineAwareGreedy", DeadlineAwareGreedy),
    "randomfit": ("RandomFit", RandomFit),
    "spf": ("ShortestPathFirst", ShortestPathFirst),
    "glb": ("GreedyGLB", GreedyGLB)
}


def _add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train-dir", default=TRAIN_DIR)
    parser.add_argument("--model-dir", default="models/hrl_final")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--ll-pretrained", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of evaluation runs per test file (for averaging results)")

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


def _generate_data(topology, distribution, difficulty, scale, requests,
                   num_files, output_dir, seed_offset=0):
    cmd = [
        sys.executable, "-u", GENERATE_SCRIPT,
        "--topology", topology,
        "--distribution", distribution,
        "--difficulty", difficulty,
        "--scale", str(scale),
        "--num-files", str(num_files),
        "--requests", str(requests),
        "--seed-offset", str(seed_offset),
        "--output", output_dir,
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    if result.returncode != 0:
        print(f"[WARN] Command failed: {' '.join(cmd)}", flush=True)
    return result.returncode == 0


def run_generate(args):
    print("\n[Generating] Topology={} Distribution={} Difficulty={}".format(
        args.topology, args.distribution, args.difficulty))
    _generate_data(args.topology, args.distribution, args.difficulty,
                   args.scale, args.requests, args.num_test_files, TEST_DIR)

def run_pretrain(args):
    train_dir = os.path.abspath(getattr(args, "train_dir", TRAIN_DIR))
    if not get_data_files(train_dir):
        print(f"[ERROR] No JSON files in {train_dir}. Run --mode generate first.")
        return
    logger = TrainingLogger(log_dir=os.path.join(ROOT_DIR, "logs/pretrain"))
    ok = _run_pretrain_inline(args, train_dir, 100, logger=logger)
    logger.save()
    logger.plot_learning_curves()
    print("[Pretrain] Complete." if ok else "[Pretrain] Failed.", flush=True)

def run_train(args):
    print("\n=== TRAINING ===")
    ll_path = getattr(args, "ll_pretrained", None)
    if not ll_path:
        candidate = os.path.join(ROOT_DIR, config.PLACER_DIR, config.PLACER_WEIGHTS_FILE)
        ll_path = candidate if os.path.exists(candidate) else None
    logger = TrainingLogger(log_dir=os.path.join(ROOT_DIR, "logs/train"))
    _run_train(args.episodes, ll_path,
               os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
               os.path.abspath(getattr(args, "train_dir", TRAIN_DIR)),
               train_request_pct=100,
               logger=logger)
    logger.save()
    logger.plot_learning_curves()

def run_eval(args):
    print("\n=== EVALUATION ===")
    _run_eval(
        os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
        os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR),
        sample_n=None,
        sample_seed=None,
        num_runs=getattr(args, "num_runs", 1),
    )

def run_baselines(args=None):
    baselines_to_run = getattr(args, "baselines", None) or list(BASELINE_REGISTRY.keys())
    plot_out = getattr(args, "plot_out", None)
    test_dir = os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR)
    csv_out = getattr(args, "csv_out", None)

    all_files = get_data_files(test_dir) or get_data_files(os.path.join(ROOT_DIR, "data"))
    if not all_files:
        print("[ERROR] No test files found. Run --mode generate first.")
        return

    files = all_files

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
            t_start = time.time()
            env = load_env_from_json(fp)
            env.set_strategy(cls(env))
            env.run_simulation()
            env.print_statistics()
            t_elapsed = time.time() - t_start
            s = env.stats
            row = {
                "algorithm": label,
                "file": os.path.basename(fp),
                "acceptance_ratio": round(s.get("acceptance_ratio", 0.0), 4),
                "accepted": s.get("accepted_requests", 0),
                "rejected": s.get("rejected_requests", 0),
                "total_cost": round(s.get("total_cost", 0.0), 2),
                "avg_cost": round(s.get("total_cost", 0.0) / max(s.get("accepted_requests", 1), 1), 2),
                "total_delay": round(s.get("total_delay", 0.0), 2),
                "computing_time": round(t_elapsed, 3),
            }
            csv_rows.append(row)
            if label not in agg:
                agg[label] = {"ar": [], "cost": [], "delay": [], "time": []}
            agg[label]["ar"].append(row["acceptance_ratio"])
            agg[label]["cost"].append(row["total_cost"])
            agg[label]["delay"].append(row["total_delay"])
            agg[label]["time"].append(row["computing_time"])
            print(f"[{label}] acceptance ratio {row['acceptance_ratio']}  completed in {t_elapsed:.3f}s")

    plot_results = []
    for label, vals in agg.items():
        plot_results.append({
            "name": label,
            "ar": float(np.mean(vals["ar"])),
            "cost": float(np.mean(vals["cost"])),
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
    save_csv(csv_rows, out, fieldnames=["algorithm", "file", "acceptance_ratio", "accepted",
                                        "rejected", "total_cost", "avg_cost", "total_delay", "computing_time"])

    import config
    model_dir = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))
    drl_weights = [config.PLACER_WEIGHTS_FILE, config.VGAE_WEIGHTS_FILE]
    if os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, w)) for w in drl_weights
    ):
        hrl_rows = []
        hrl_agg = {"ar": [], "cost": [], "delay": [], "time": []}
        for fp in files:
            filename = os.path.basename(fp)
            print(f"\n[DRL-NFV] Evaluating {filename} from {model_dir} ...")
            t_start = time.time()
            env = load_env_from_json(fp)
            strategy = DRL_Strategy(env, is_training=False, episodes=1)
            strategy.load_model(model_dir)
            env.set_strategy(strategy)
            drl_stats = strategy.run_simulation_eval()
            env.print_statistics()
            t_elapsed = time.time() - t_start
            row = {
                "algorithm": "DRL-NFV",
                "file": filename,
                "acceptance_ratio": round(drl_stats.get("acceptance_ratio", 0.0), 4),
                "accepted": drl_stats.get("accepted_requests", 0),
                "rejected": drl_stats.get("rejected_requests", 0),
                "total_cost": round(drl_stats.get("total_cost", 0.0), 2),
                "avg_cost": round(drl_stats.get("average_cost", 0.0), 2),
                "total_delay": round(drl_stats.get("total_delay", 0.0), 2),
                "computing_time": round(t_elapsed, 3),
            }
            csv_rows.append(row)
            hrl_rows.append({"name": "DRL-NFV", "ar": row["acceptance_ratio"],
                             "cost": row["total_cost"], "delay": row["total_delay"], "time": row["computing_time"]})
            for k in ["ar", "cost", "delay", "time"]:
                hrl_agg[k].append(hrl_rows[-1][k])
            print(f"[DRL-NFV] {filename:<40} completed in {t_elapsed:.3f}s")

        drl_plot = [{"name": "DRL-NFV",
                     "ar": float(np.mean(hrl_agg["ar"])),
                     "cost": float(np.mean(hrl_agg["cost"])),
                     "delay": float(np.mean(hrl_agg["delay"]))}]
        cmp_out = (plot_out.replace(".png", "_vs_drl.png") if plot_out
                   else os.path.join(ROOT_DIR, "drl_vs_baselines.png"))
        _plot_eval_vs_baselines(drl_plot, plot_results, out_path=cmp_out)

        save_csv(csv_rows, out, fieldnames=["algorithm", "file", "acceptance_ratio", "accepted",
                                             "rejected", "total_cost", "avg_cost", "total_delay", "computing_time"])


def main():
    p = argparse.ArgumentParser(description="NFV VNF Placement – DRL-NFV")
    p.add_argument("--mode", default="baseline",
                   choices=["generate", "pretrain", "train", "eval", "baseline"])

    _add_data_generation_args(p)
    _add_shared_args(p)
    _add_training_budget_args(p)

    p.add_argument("--baselines", nargs="+", default=None,
                   choices=list(BASELINE_REGISTRY.keys()))
    p.add_argument("--plot-out", type=str, default=None)

    args = p.parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "pretrain":
        run_pretrain(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "baseline":
        run_baselines(args)


if __name__ == "__main__":
    main()