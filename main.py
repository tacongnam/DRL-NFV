import os, sys, argparse, time
import numpy as np
import config
from strategy import GreedyFIFS, BestFit, DeadlineAwareGreedy, RandomFit, ShortestPathFirst, GreedyGLB, DRL_Strategy
from data.load_data import load_env_from_json, get_data_files, save_csv
from utils import _run_eval, _run_train, _run_pretrain_inline, _plot_baseline_results, _plot_eval_vs_baselines, TrainingLogger
from utils.metrics import ExperimentTracker

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

for _d in ["models/vgae_pretrained", "models/placer", "models/hrl_final"]:
    os.makedirs(os.path.join(ROOT_DIR, _d), exist_ok=True)

sys.path.insert(0, ROOT_DIR)

TRAIN_DIR = os.path.join(ROOT_DIR, "data/train")
TEST_DIR  = os.path.join(ROOT_DIR, "data/test")
DEFAULT_EPISODES = 100
BASELINE_REGISTRY = {
    "fifs":      ("GreedyFIFS",           GreedyFIFS),
    "bestfit":   ("BestFit",              BestFit),
    "deadline":  ("DeadlineAwareGreedy",  DeadlineAwareGreedy),
    "randomfit": ("RandomFit",            RandomFit),
    "spf":       ("ShortestPathFirst",    ShortestPathFirst),
    "glb":       ("GreedyGLB",            GreedyGLB),
}

def _add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train-dir",    default=TRAIN_DIR)
    parser.add_argument("--model-dir",    default="models/hrl_final")
    parser.add_argument("--test-dir",     default=TEST_DIR)
    parser.add_argument("--ll-pretrained", type=str, default=None)
    parser.add_argument("--num-runs",      type=int, default=1)

def _add_training_budget_args(parser: argparse.ArgumentParser):
    parser.add_argument("--episodes",    type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--vgae-epochs", type=int, default=150)
    parser.add_argument("--ll-episodes", type=int, default=200)

def _collect_baseline_stats(fp: str, key: str, cls) -> dict:
    label    = BASELINE_REGISTRY[key][0]
    t_start  = time.time()
    env      = load_env_from_json(fp)
    env.set_strategy(cls(env))
    env.run_simulation()
    env.print_statistics()
    elapsed  = time.time() - t_start
    s        = env.stats

    total_rev  = s.get("total_revenue",  0.0)
    total_cost = max(s.get("total_cost", 1e-6), 1e-6)
    duration   = max(s.get("simulation_duration", 1.0), 1.0)

    return {
        "algorithm":        label,
        "file":             os.path.basename(fp),
        "acceptance_ratio": round(s.get("acceptance_ratio", 0.0), 4),
        "accepted":         s.get("accepted_requests", 0),
        "rejected":         s.get("rejected_requests", 0),
        "total_cost":       round(total_cost, 2),
        "avg_cost":         round(total_cost / max(s.get("accepted_requests", 1), 1), 2),
        "total_delay":      round(s.get("total_delay", 0.0), 2),
        "ltr":              round(total_rev / duration, 4),
        "lt_r2c":           round(total_rev / total_cost, 4),
        "computing_time":   round(elapsed, 3),
    }

def run_pretrain(args):
    print("\n=== PRE-TRAINING ===")
    train_dir = os.path.abspath(getattr(args, "train_dir", TRAIN_DIR))
    if not get_data_files(train_dir):
        print(f"[ERROR] No JSON files in {train_dir}.")
        return
    logger = TrainingLogger(log_dir=os.path.join(ROOT_DIR, "logs/pretrain"))
    ok = _run_pretrain_inline(args, train_dir, logger=logger)
    logger.save()
    logger.plot_learning_curves()
    print("[Pretrain] Complete." if ok else "[Pretrain] Failed.", flush=True)

def run_train(args):
    print("\n=== TRAINING ===")
    ll_path = getattr(args, "ll_pretrained", None)
    if not ll_path:
        candidate = os.path.join(ROOT_DIR, config.PLACER_DIR, config.PLACER_WEIGHTS_FILE)
        ll_path   = candidate if os.path.exists(candidate) else None
    logger = TrainingLogger(log_dir=os.path.join(ROOT_DIR, "logs/train"))
    _run_train(args.episodes, ll_path,
               os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
               os.path.abspath(getattr(args, "train_dir", TRAIN_DIR)),
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
    plot_out  = getattr(args, "plot_out",  None)
    test_dir  = os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR)
    csv_out   = getattr(args, "csv_out",   None)
    tracker   = ExperimentTracker()

    files = get_data_files(test_dir) or get_data_files(os.path.join(ROOT_DIR, "data"))
    if not files:
        print("[ERROR] No test files found.")
        return

    print(f"\nBaseline comparison on {len(files)} file(s).")

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
            row = _collect_baseline_stats(fp, key, cls)
            csv_rows.append(row)
            if label not in agg:
                agg[label] = {"ar": [], "cost": [], "delay": [], "ltr": [], "r2c": [], "time": []}
            agg[label]["ar"].append(row["acceptance_ratio"])
            agg[label]["cost"].append(row["total_cost"])
            agg[label]["delay"].append(row["total_delay"])
            agg[label]["ltr"].append(row["ltr"])
            agg[label]["r2c"].append(row["lt_r2c"])
            agg[label]["time"].append(row["computing_time"])

    plot_results = []
    for label, vals in agg.items():
        entry = {
            "name":   label,
            "ar":     float(np.mean(vals["ar"])),
            "cost":   float(np.mean(vals["cost"])),
            "delay":  float(np.mean(vals["delay"])),
            "ltr":    float(np.mean(vals["ltr"])),
            "lt_r2c": float(np.mean(vals["r2c"])),
        }
        plot_results.append(entry)
        tracker.record_algo(label, {
            "acceptance_ratio": entry["ar"],
            "ltr":              entry["ltr"],
            "lt_r2c":           entry["lt_r2c"],
            "accepted_requests": int(np.sum(vals["ar"])),
            "rejected_requests": 0,
        })

    if len(plot_results) > 1:
        print("\n=== BASELINE SUMMARY (avg across files) ===")
        print(f"{'Algorithm':<25} {'AR':>8} {'LTR':>10} {'LT-R2C':>10} {'Cost':>10}")
        print("-" * 65)
        for r in plot_results:
            print(f"{r['name']:<25} {r['ar']:>8.3f} {r['ltr']:>10.2f} {r['lt_r2c']:>10.3f} {r['cost']:>10.1f}")
        _plot_baseline_results(plot_results, out_path=plot_out)

    out = csv_out or os.path.join(ROOT_DIR, "baseline_results.csv")
    save_csv(csv_rows, out,
             fieldnames=["algorithm", "file", "acceptance_ratio", "accepted",
                         "rejected", "total_cost", "avg_cost", "total_delay",
                         "ltr", "lt_r2c", "computing_time"])

    model_dir   = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))
    drl_weights = [config.PLACER_WEIGHTS_FILE, config.VGAE_WEIGHTS_FILE]
    if os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, w)) for w in drl_weights
    ):
        _run_drl_comparison(args, files, csv_rows, plot_results, plot_out, out,
                            model_dir, tracker)

    if getattr(args, "ablation", False):
        run_admission_ablation(args, files, tracker)

    if getattr(args, "arrival_rate_exp", False):
        run_arrival_rate_experiment(args, files, tracker)

    tracker.print_comparison_table()

def _run_drl_comparison(args, files, csv_rows, plot_results, plot_out, csv_out,
                        model_dir, tracker: ExperimentTracker):
    hrl_agg = {"ar": [], "cost": [], "delay": [], "ltr": [], "r2c": [], "time": []}
    for fp in files:
        filename = os.path.basename(fp)
        t_start  = time.time()
        env      = load_env_from_json(fp)
        strategy = DRL_Strategy(env, is_training=False, episodes=1)
        strategy.load_model(model_dir)
        env.set_strategy(strategy)
        drl_stats = strategy.run_simulation_eval()
        env.print_statistics()
        elapsed = time.time() - t_start

        total_rev  = drl_stats.get("total_revenue",  0.0)
        total_cost = max(drl_stats.get("total_cost", 1e-6), 1e-6)
        duration   = max(drl_stats.get("simulation_duration", 1.0), 1.0)

        row = {
            "algorithm":        "DRL-NFV",
            "file":             filename,
            "acceptance_ratio": round(drl_stats.get("acceptance_ratio", 0.0), 4),
            "accepted":         drl_stats.get("accepted_requests", 0),
            "rejected":         drl_stats.get("rejected_requests", 0),
            "total_cost":       round(total_cost, 2),
            "avg_cost":         round(drl_stats.get("average_cost", 0.0), 2),
            "total_delay":      round(drl_stats.get("total_delay", 0.0), 2),
            "ltr":              round(drl_stats.get("ltr", total_rev / duration), 4),
            "lt_r2c":           round(drl_stats.get("lt_r2c", total_rev / total_cost), 4),
            "computing_time":   round(elapsed, 3),
        }
        csv_rows.append(row)
        for k, src in [("ar", "acceptance_ratio"), ("cost", "total_cost"),
                       ("delay", "total_delay"), ("ltr", "ltr"), ("r2c", "lt_r2c")]:
            hrl_agg[k].append(row[src])
        hrl_agg["time"].append(row["computing_time"])

    tracker.record_algo("DRL-NFV", {
        "acceptance_ratio": float(np.mean(hrl_agg["ar"])),
        "ltr":              float(np.mean(hrl_agg["ltr"])),
        "lt_r2c":           float(np.mean(hrl_agg["r2c"])),
        "accepted_requests": 0,
        "rejected_requests": 0,
    })

    drl_plot  = [{"name": "DRL-NFV",
                  "ar":     float(np.mean(hrl_agg["ar"])),
                  "cost":   float(np.mean(hrl_agg["cost"])),
                  "delay":  float(np.mean(hrl_agg["delay"])),
                  "ltr":    float(np.mean(hrl_agg["ltr"])),
                  "lt_r2c": float(np.mean(hrl_agg["r2c"]))}]
    cmp_out   = (plot_out.replace(".png", "_vs_drl.png") if plot_out
                 else os.path.join(os.path.dirname(csv_out), "drl_vs_baselines.png"))
    _plot_eval_vs_baselines(drl_plot, plot_results, out_path=cmp_out)

    save_csv(csv_rows, csv_out,
             fieldnames=["algorithm", "file", "acceptance_ratio", "accepted",
                         "rejected", "total_cost", "avg_cost", "total_delay",
                         "ltr", "lt_r2c", "computing_time"])


def run_admission_ablation(args, files: list, tracker: ExperimentTracker):
    print("\n=== Admission Control Ablation ===")
    model_dir = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))

    for variant_name, use_admission in [
        ("DRL-NoAdmission",      False),
        ("DRL-NFV (admission)",  True),
    ]:
        agg_ar, agg_ltr, agg_r2c = [], [], []
        for fp in files:
            env      = load_env_from_json(fp)
            strategy = DRL_Strategy(env, is_training=False, episodes=1)
            strategy.load_model(model_dir)

            if not use_admission:
                strategy.admission = _NoAdmissionAgent()

            env.set_strategy(strategy)
            s = strategy.run_simulation_eval()
            agg_ar.append(s.get("acceptance_ratio", 0.0))
            agg_ltr.append(s.get("ltr", 0.0))
            agg_r2c.append(s.get("lt_r2c", 0.0))

        tracker.record_algo(variant_name, {
            "acceptance_ratio": float(np.mean(agg_ar)),
            "ltr":              float(np.mean(agg_ltr)),
            "lt_r2c":           float(np.mean(agg_r2c)),
            "accepted_requests": 0, "rejected_requests": 0,
        })
        print(f"  {variant_name:<30}  AR={np.mean(agg_ar):.3f}"
              f"  LTR={np.mean(agg_ltr):.2f}  R2C={np.mean(agg_r2c):.3f}")

    tracker.print_admission_ablation()


def run_arrival_rate_experiment(args, files: list, tracker: ExperimentTracker):
    print("\n=== Arrival Rate Experiment ===")
    model_dir    = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))
    arrival_rates = getattr(args, "arrival_rates", [0.04, 0.08, 0.12, 0.16, 0.18])
    baselines_to_run = getattr(args, "baselines", None) or list(BASELINE_REGISTRY.keys())
    results_by_rate  = {}

    for rate in arrival_rates:
        results_by_rate[rate] = {}
        for key in baselines_to_run:
            if key not in BASELINE_REGISTRY:
                continue
            label, cls = BASELINE_REGISTRY[key]
            ar_list = []
            for fp in files:
                env = load_env_from_json(fp)
                _scale_arrival_rate(env, rate)
                env.set_strategy(cls(env))
                env.run_simulation()
                ar_list.append(env.stats.get("acceptance_ratio", 0.0))
            results_by_rate[rate][label] = {"acceptance_ratio": float(np.mean(ar_list))}

        ar_list = []
        for fp in files:
            env      = load_env_from_json(fp)
            _scale_arrival_rate(env, rate)
            strategy = DRL_Strategy(env, is_training=False, episodes=1)
            strategy.load_model(model_dir)
            env.set_strategy(strategy)
            s = strategy.run_simulation_eval()
            ar_list.append(s.get("acceptance_ratio", 0.0))
        results_by_rate[rate]["DRL-NFV"] = {"acceptance_ratio": float(np.mean(ar_list))}
        print(f"  rate={rate:.3f}  DRL-NFV AR={np.mean(ar_list):.3f}")

    tracker.print_arrival_rate_table(results_by_rate)


def _scale_arrival_rate(env, target_rate: float):
    if not hasattr(env, 'requests') or not env.requests:
        return
    base_intervals = []
    sorted_reqs = sorted(env.requests, key=lambda r: r.arrival_time)
    for i in range(1, len(sorted_reqs)):
        base_intervals.append(
            sorted_reqs[i].arrival_time - sorted_reqs[i - 1].arrival_time)
    if not base_intervals:
        return
    base_rate = 1.0 / max(np.mean(base_intervals), 1e-6)
    scale     = base_rate / max(target_rate, 1e-6)
    t0        = sorted_reqs[0].arrival_time
    for req in sorted_reqs:
        req.arrival_time = t0 + (req.arrival_time - t0) * scale
        req.end_time     = req.arrival_time + (req.end_time - req.arrival_time)


class _NoAdmissionAgent:
    def decide(self, gp, gq, oq, training=False):
        return True, 0.0, 0.0

    def push_triplet(self, *a): pass
    def reset_history(self): pass
    def _get_window(self): return np.zeros((1, 10, 18), np.float32)
    def record(self, *a): pass
    def train_ppo(self, **kw): return None
    def save_weights(self, d): pass
    def load_weights(self, d): pass


def main():
    p = argparse.ArgumentParser(description="NFV VNF Placement – DRL-NFV")
    p.add_argument("--mode", default="baseline",
                   choices=["generate", "pretrain", "train", "eval",
                             "baseline", "ablation", "arrival_rate"])

    _add_shared_args(p)
    _add_training_budget_args(p)

    p.add_argument("--baselines",      nargs="+", default=None,
                   choices=list(BASELINE_REGISTRY.keys()))
    p.add_argument("--plot-out",       type=str,   default=None)
    p.add_argument("--ablation",       action="store_true",
                   help="Run admission control ablation study")
    p.add_argument("--arrival-rate-exp", action="store_true",
                   help="Run arrival rate sensitivity experiment")
    p.add_argument("--arrival-rates",  nargs="+", type=float,
                   default=[0.04, 0.08, 0.12, 0.16, 0.18])
    args = p.parse_args()

    if args.mode == "pretrain":
        run_pretrain(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "ablation":
        args.ablation = True
        files = get_data_files(os.path.abspath(args.test_dir))
        tracker = ExperimentTracker()
        run_admission_ablation(args, files, tracker)
    elif args.mode == "arrival_rate":
        args.arrival_rate_exp = True
        files = get_data_files(os.path.abspath(args.test_dir))
        tracker = ExperimentTracker()
        run_arrival_rate_experiment(args, files, tracker)
    elif args.mode == "baseline":
        run_baselines(args)


if __name__ == "__main__":
    main()