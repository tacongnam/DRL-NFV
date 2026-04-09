import os, sys, json, argparse, subprocess

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for _d in ["models/vgae_pretrained", "models/ll_pretrained",
           "models/hrl_final", "data/train", "data/test"]:
    os.makedirs(_d, exist_ok=True)

from env.vnf      import VNF, ListOfVnfs
from env.request  import Request, ListOfRequests
from env.network  import Network
from env.env      import Env
from strategy.fifs          import GreedyFIFS
from strategy.glb           import GreedyGLB
from strategy.spf           import ShortestPathFirst
from strategy.best_fit      import BestFit
from strategy.deadline_aware import DeadlineAwareGreedy
from strategy.random_fit    import RandomFit
from strategy.hrl           import HRL_VGAE_Strategy

TRAIN_DIR        = "data/train"
TEST_DIR         = "data/test"
DEFAULT_EPISODES = 300

BASELINE_REGISTRY = {
    "fifs":     ("GreedyFIFS",           GreedyFIFS),
    "glb":      ("GreedyGLB",            GreedyGLB),
    "spf":      ("ShortestPathFirst",    ShortestPathFirst),
    "bestfit":  ("BestFit",              BestFit),
    "deadline": ("DeadlineAwareGreedy",  DeadlineAwareGreedy),
    "random":   ("RandomFit",            RandomFit),
}


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
                capacity={"mem": nd.get("h_v", 1.), "cpu": nd.get("c_v", 1.), "ram": nd.get("r_v", 1.)},
                cost={"mem": nd.get("cost_h", 1.), "cpu": nd.get("cost_c", 1.), "ram": nd.get("cost_r", 1.)})
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]),
                         lnk.get("b_l", 1.), lnk.get("d_l", 1.))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f", 1.), c_f=vd.get("c_f", 1.), r_f=vd.get("r_f", 1.),
                         d_f={k: v for k, v in vd.get("d_f", {}).items()}))

    for idx, rd in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T", 0),
            delay_max=rd.get("d_max", 100.),
            start_node=str(rd.get("st_r", "")), end_node=str(rd.get("d_r", "")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r", [])],
            bandwidth=rd.get("b_r", 1.)))
    return Env(network, vnfs, requests)


def get_data_files(d: str):
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    return []


def _plot_baseline_results(results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg" if out_path else "TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[Plot] matplotlib not available, skipping chart.")
        return

    names  = [r["name"] for r in results]
    ar     = [r["ar"] for r in results]
    cost   = [r["cost"] for r in results]
    delay  = [r["delay"] for r in results]

    x      = np.arange(len(names))
    width  = 0.55
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Baseline Algorithm Comparison", fontsize=14, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(x, ar, width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Acceptance Ratio")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, ar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, cost, width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Total Cost")
    ax.set_ylabel("Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, cost):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cost) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    bars = ax.bar(x, delay, width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Total Delay")
    ax.set_ylabel("Delay")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, delay):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(delay) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {out_path}")
    else:
        try:
            plt.show()
        except Exception:
            save_fallback = "baseline_comparison.png"
            plt.savefig(save_fallback, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved → {save_fallback}")
    plt.close()


def _plot_eval_vs_baselines(hrl_results: list, baseline_results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg" if out_path else "TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    all_results = baseline_results + hrl_results
    names  = [r["name"] for r in all_results]
    ar     = [r["ar"] for r in all_results]
    cost   = [r["cost"] for r in all_results]

    x      = np.arange(len(names))
    width  = 0.55
    n_base = len(baseline_results)
    colors = (["#5b8dd9"] * n_base) + (["#e05c5c"] * len(hrl_results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HRL-VGAE vs Baselines", fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], ar,   "Acceptance Ratio", "Ratio"),
        (axes[1], cost, "Total Cost",       "Cost"),
    ]:
        bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}" if title == "Acceptance Ratio" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)

    import matplotlib.patches as mpatches
    axes[0].legend(handles=[
        mpatches.Patch(color="#5b8dd9", label="Baseline"),
        mpatches.Patch(color="#e05c5c", label="HRL-VGAE"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {out_path}")
    else:
        try:
            plt.show()
        except Exception:
            save_fallback = "hrl_vs_baselines.png"
            plt.savefig(save_fallback, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved → {save_fallback}")
    plt.close()


def run_pipeline(args):
    print("\n" + "="*60)
    print("HRL-VGAE PIPELINE: generate → pretrain → train → eval")
    print("="*60)

    print("\n[1/4] Generating data ...")
    for diff, offset in [("easy", 0), ("hard", args.num_train_files)]:
        subprocess.run([sys.executable, "data/generate.py",
            "--topology", args.topology, "--distribution", args.distribution,
            "--difficulty", diff, "--scale", str(args.scale),
            "--num-files", str(args.num_train_files),
            "--requests", str(args.requests),
            "--seed-offset", str(offset), "--output", TRAIN_DIR])

    subprocess.run([sys.executable, "data/generate.py",
        "--topology", args.topology, "--distribution", args.distribution,
        "--difficulty", args.difficulty, "--scale", str(args.scale),
        "--num-files", str(args.num_test_files),
        "--requests", str(args.requests), "--output", TEST_DIR])
    print(f"  train={len(get_data_files(TRAIN_DIR))}  test={len(get_data_files(TEST_DIR))}")

    print("\n[2/4] Pre-training VGAE + LL-DQN ...")
    subprocess.run([sys.executable, "models/pretrain.py",
        "--phase", "both", "--train-dir", TRAIN_DIR,
        "--vgae-epochs", str(args.vgae_epochs),
        "--ll-episodes", str(args.ll_episodes)])

    print("\n[3/4] Training HRL ...")
    _run_train(args.episodes, "models/ll_pretrained/ll_dqn_weights.weights.h5",
               "models/hrl_final", TRAIN_DIR)

    print("\n[4/4] Evaluating ...")
    _run_eval("models/hrl_final", TEST_DIR)

    print("\n" + "="*60 + "\nPIPELINE COMPLETE\n" + "="*60)


def run_generate(args):
    print("\n=== DATA GENERATION ===")
    for dest, n_files in [(TRAIN_DIR, args.num_train_files), (TEST_DIR, args.num_test_files)]:
        if n_files <= 0:
            continue
        subprocess.run([sys.executable, "data/generate.py",
            "--topology", args.topology, "--distribution", args.distribution,
            "--difficulty", args.difficulty, "--scale", str(args.scale),
            "--num-files", str(n_files), "--requests", str(args.requests),
            "--output", dest])
    print(f"train={len(get_data_files(TRAIN_DIR))}  test={len(get_data_files(TEST_DIR))}")


def run_pretrain(args):
    subprocess.run([sys.executable, "models/pretrain.py",
        "--phase", "both",
        "--train-dir", getattr(args, "train_dir", TRAIN_DIR),
        "--vgae-epochs", str(getattr(args, "vgae_epochs", 200)),
        "--ll-episodes", str(getattr(args, "ll_episodes", 200))])


def _run_train(episodes, ll_pretrained, save_dir, train_dir):
    files = get_data_files(train_dir)
    if not files:
        print(f"No training files in {train_dir}")
        return None

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


def _run_eval(model_dir, test_dir, test_files=None):
    files = test_files or get_data_files(test_dir)
    if not files:
        print("No test files found.")
        return []

    results = []
    for fp in files:
        print(f"\n--- {os.path.basename(fp)} ---")
        env = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
        if model_dir:
            strategy.load_model(model_dir)
        env.set_strategy(strategy)
        stats = strategy.run_simulation_eval()
        env.print_statistics()
        results.append({
            "name": "HRL-VGAE",
            "file": os.path.basename(fp),
            "ar":   stats.get("acceptance_ratio", 0),
            "acc":  stats.get("accepted_requests", 0),
            "rej":  stats.get("rejected_requests", 0),
            "cost": stats.get("total_cost", 0),
            "delay": stats.get("total_delay", 0),
        })

    print("\n=== EVAL SUMMARY ===")
    print(f"{'File':<35} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<35} {r['ar']:>9.3f} {r['acc']:>6} {r['rej']:>6} {r['cost']:>10.1f}")
    if results:
        avg_ar = sum(r["ar"] for r in results) / len(results)
        print(f"\nAverage acceptance ratio: {avg_ar:.3f}")
    return results


def run_eval(args):
    print("\n=== EVALUATION ===")
    _run_eval(getattr(args, "model_dir", None),
              getattr(args, "test_dir", None) or TEST_DIR,
              getattr(args, "test_files", None))


def run_baselines(args=None):
    baselines_to_run = getattr(args, "baselines", None) or list(BASELINE_REGISTRY.keys())
    plot_out         = getattr(args, "plot_out", None)
    test_dir         = getattr(args, "test_dir", None) or TEST_DIR

    files = get_data_files(test_dir) or get_data_files("data")
    if not files:
        print("No test files found.")
        return

    fp = files[0]
    print(f"\nBaseline comparison on: {os.path.basename(fp)}")

    results = []
    for key in baselines_to_run:
        if key not in BASELINE_REGISTRY:
            print(f"  [WARN] Unknown baseline '{key}', skipping.")
            continue
        label, cls = BASELINE_REGISTRY[key]
        print(f"\n[{label}]")
        env = load_env_from_json(fp)
        strategy = cls(env)
        env.set_strategy(strategy)
        env.run_simulation()
        env.print_statistics()
        s = env.stats
        results.append({
            "name":  label,
            "ar":    s.get("acceptance_ratio", 0.0),
            "acc":   s.get("accepted_requests", 0),
            "rej":   s.get("rejected_requests", 0),
            "cost":  s.get("total_cost", 0.0),
            "delay": s.get("total_delay", 0.0),
        })

    if len(results) > 1:
        print("\n=== BASELINE SUMMARY ===")
        print(f"{'Algorithm':<25} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10} {'Delay':>10}")
        print("-"*70)
        for r in results:
            print(f"{r['name']:<25} {r['ar']:>9.3f} {r['acc']:>6} {r['rej']:>6} "
                  f"{r['cost']:>10.1f} {r['delay']:>10.1f}")
        _plot_baseline_results(results, out_path=plot_out)

    model_dir = getattr(args, "model_dir", "models/hrl_final")
    if os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, w))
        for w in ["hl_pmdrl_weights.weights.h5", "ll_dqn_weights.weights.h5"]
    ):
        print(f"\n[HRL-VGAE] Evaluating trained model from {model_dir} ...")
        env = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
        strategy.load_model(model_dir)
        env.set_strategy(strategy)
        hrl_stats = strategy.run_simulation_eval()
        env.print_statistics()
        hrl_result = [{
            "name":  "HRL-VGAE",
            "ar":    hrl_stats.get("acceptance_ratio", 0.0),
            "cost":  hrl_stats.get("total_cost", 0.0),
            "delay": hrl_stats.get("total_delay", 0.0),
        }]
        cmp_out = plot_out.replace(".png", "_vs_hrl.png") if plot_out else None
        _plot_eval_vs_baselines(hrl_result, results, out_path=cmp_out)


def main():
    p = argparse.ArgumentParser(description="NFV VNF Placement – HRL-VGAE")
    p.add_argument("--mode", default="baseline",
                   choices=["pipeline", "generate", "pretrain", "train", "eval", "baseline"])

    p.add_argument("--topology",        default="nsf",
                   choices=["nsf", "conus", "cogent"])
    p.add_argument("--distribution",    default="rural",
                   choices=["uniform", "rural", "urban", "centers"])
    p.add_argument("--difficulty",      default="easy",
                   choices=["easy", "hard"])
    p.add_argument("--scale",           type=int, default=50)
    p.add_argument("--requests",        type=int, default=50)
    p.add_argument("--num-train-files", type=int, default=5)
    p.add_argument("--num-test-files",  type=int, default=3)

    p.add_argument("--episodes",        type=int, default=DEFAULT_EPISODES)
    p.add_argument("--ll-pretrained",   type=str, default=None)
    p.add_argument("--model-dir",       default="models/hrl_final")
    p.add_argument("--train-dir",       default=TRAIN_DIR)
    p.add_argument("--test-dir",        default=None)
    p.add_argument("--test-files",      nargs="+", default=None)

    p.add_argument("--vgae-epochs",     type=int, default=200)
    p.add_argument("--ll-episodes",     type=int, default=200)

    p.add_argument("--baselines",       nargs="+", default=None,
                   choices=list(BASELINE_REGISTRY.keys()),
                   help="Baselines to run (default: all)")
    p.add_argument("--plot-out",        type=str, default=None,
                   help="Save plot to file instead of showing")

    args = p.parse_args()

    if args.mode == "pipeline":   run_pipeline(args)
    elif args.mode == "generate": run_generate(args)
    elif args.mode == "pretrain": run_pretrain(args)
    elif args.mode == "train":    run_train(args)
    elif args.mode == "eval":     run_eval(args)
    elif args.mode == "baseline": run_baselines(args)


if __name__ == "__main__":
    main()