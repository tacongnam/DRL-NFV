import os, sys, json, argparse, subprocess, math, random, csv
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

for _d in ["models/vgae_pretrained", "models/ll_pretrained",
           "models/hrl_final", "data/train", "data/test"]:
    os.makedirs(os.path.join(ROOT_DIR, _d), exist_ok=True)

sys.path.insert(0, ROOT_DIR)

from env.vnf      import VNF, ListOfVnfs
from env.request  import Request, ListOfRequests
from env.network  import Network
from env.env      import Env
from strategy.fifs           import GreedyFIFS
from strategy.glb            import GreedyGLB
from strategy.spf            import ShortestPathFirst
from strategy.best_fit       import BestFit
from strategy.deadline_aware import DeadlineAwareGreedy
from strategy.random_fit     import RandomFit
from strategy.hrl            import HRL_VGAE_Strategy

TRAIN_DIR        = os.path.join(ROOT_DIR, "data/train")
TEST_DIR         = os.path.join(ROOT_DIR, "data/test")
GENERATE_SCRIPT  = os.path.join(ROOT_DIR, "data/generate.py")
DEFAULT_EPISODES = 60
DEFAULT_TRAIN_REQUEST_PCT = 0
DEFAULT_PRETRAIN_REQUEST_PCT = 0

BASELINE_REGISTRY = {
    "fifs":     ("GreedyFIFS",          GreedyFIFS),
    #"glb":      ("GreedyGLB",           GreedyGLB),
    #"spf":      ("ShortestPathFirst",   ShortestPathFirst),
    "bestfit":  ("BestFit",             BestFit),
    "deadline": ("DeadlineAwareGreedy", DeadlineAwareGreedy),
    #"random":   ("RandomFit",           RandomFit),
}


def _sample_requests(req_rows: list, request_pct: int = 0) -> list:
    req_limit = _resolve_request_limit(len(req_rows), request_pct=request_pct)
    if req_limit is None or req_limit <= 0 or len(req_rows) <= req_limit:
        return req_rows
    idxs = np.linspace(0, len(req_rows) - 1, num=req_limit, dtype=int)
    return [req_rows[i] for i in idxs]


def _resolve_request_limit(total_requests: int, request_pct: int = 0) -> int | None:
    if request_pct is None or request_pct <= 0:
        return None
    return max(1, math.ceil(total_requests * request_pct / 100.0))


def load_env_from_json(filepath: str, request_pct: int = 0) -> Env:
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

    req_rows = sorted(data.get("R", []), key=lambda r: r.get("T", 0))
    req_rows = _sample_requests(req_rows, request_pct=request_pct)

    for idx, rd in enumerate(req_rows):
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


def _sample_files(files: list, n: int | None, seed: int | None = None) -> list:
    """Randomly sample n files from files list. Returns all if n is None or >= len."""
    if not files or n is None or n <= 0 or n >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, n))


def _print_selected_files(label: str, files: list, request_pct: int = 0):
    print(f"\n[{label}] Selected {len(files)} file(s)")
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        total_requests = len(data.get("R", []))
        req_limit = _resolve_request_limit(total_requests, request_pct=request_pct)
        req_label = total_requests if req_limit is None else min(total_requests, req_limit)
        print(f"  - {os.path.basename(fp)}: req={req_label}/{total_requests}")


def _save_csv(results: list, path: str, fieldnames: list = None):
    if not results:
        return
    fieldnames = fieldnames or list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[CSV] Saved → {path}")


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
        ("--difficulty", "easy", ["easy", "hard"]),
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


def _run_subprocess(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    if result.returncode != 0:
        print(f"[WARN] Command failed: {' '.join(str(c) for c in cmd)}", flush=True)
    return result.returncode == 0


def _python_cmd(script_path: str, *args: str):
    return [sys.executable, "-u", script_path, *args]


def _run_pretrain_inline(args, train_dir: str):
    from models import pretrain as pretrain_module

    selected = pretrain_module.get_train_files(train_dir)
    if not selected:
        print("[Pretrain] No training files selected.", flush=True)
        return False

    req_pct = getattr(args, "pretrain_request_pct", DEFAULT_PRETRAIN_REQUEST_PCT)
    pretrain_module.print_selected_files(selected, req_pct)

    print(f"[Pretrain] Running inline on {train_dir}", flush=True)
    vgae = None
    vgae = pretrain_module.pretrain_vgae(
        selected,
        epochs=getattr(args, "vgae_epochs", 60),
        request_pct=req_pct,
    )
    if vgae is None and getattr(args, "ll_episodes", 0) > 0:
        vgae_path = os.path.join(ROOT_DIR, "models/vgae_pretrained", "vgae_weights.npy")
        if os.path.exists(vgae_path):
            vgae = pretrain_module.VGAENetwork(latent_dim=pretrain_module.LATENT_DIM)
            vgae.load_weights(vgae_path)

    if vgae is not None:
        pretrain_module.pretrain_ll(
            selected,
            vgae,
            episodes=getattr(args, "ll_episodes", 60),
            request_pct=req_pct,
        )
    else:
        print("[Pretrain] Skipped LL pretrain because VGAE was not produced.", flush=True)

    vgae_out = os.path.join(ROOT_DIR, "models/vgae_pretrained", "vgae_weights.npy")
    ll_out = os.path.join(ROOT_DIR, "models/ll_pretrained", "ll_dqn_weights.npy")
    print(f"[Pretrain] VGAE saved: {os.path.exists(vgae_out)} -> {vgae_out}", flush=True)
    print(f"[Pretrain] LL saved: {os.path.exists(ll_out)} -> {ll_out}", flush=True)
    return os.path.exists(vgae_out) or os.path.exists(ll_out)


def _generate_data(topology, distribution, difficulty, scale, requests,
                   num_files, output_dir, seed_offset=0):
    return _run_subprocess(_python_cmd(
        GENERATE_SCRIPT,
        "--topology",    topology,
        "--distribution", distribution,
        "--difficulty",  difficulty,
        "--scale",       str(scale),
        "--num-files",   str(num_files),
        "--requests",    str(requests),
        "--seed-offset", str(seed_offset),
        "--output",      output_dir,
    ))


def _plot_baseline_results(results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plot] matplotlib not available, skipping.")
        return

    names = [r["name"]  for r in results]
    ar    = [r["ar"]    for r in results]
    cost  = [r["cost"]  for r in results]
    delay = [r["delay"] for r in results]
    x     = np.arange(len(names))
    w     = 0.55
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Baseline Algorithm Comparison", fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], ar,    "Acceptance Ratio", "Ratio"),
        (axes[1], cost,  "Total Cost",       "Cost"),
        (axes[2], delay, "Total Delay",      "Delay"),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        mx = max(vals) if max(vals) > 0 else 1
        fmt = ".3f" if title == "Acceptance Ratio" else ".1f"
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_to = out_path or os.path.join(ROOT_DIR, "baseline_comparison.png")
    plt.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_to}")
    plt.close()


def _plot_eval_vs_baselines(hrl_results: list, baseline_results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return

    all_r  = baseline_results + hrl_results
    names  = [r["name"] for r in all_r]
    ar     = [r["ar"]   for r in all_r]
    cost   = [r["cost"] for r in all_r]
    x      = np.arange(len(names))
    w      = 0.55
    n_base = len(baseline_results)
    colors = ["#5b8dd9"] * n_base + ["#e05c5c"] * len(hrl_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HRL-VGAE vs Baselines", fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], ar,   "Acceptance Ratio", "Ratio"),
        (axes[1], cost, "Total Cost",       "Cost"),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        mx  = max(vals) if max(vals) > 0 else 1
        fmt = ".3f" if title == "Acceptance Ratio" else ".1f"
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)

    axes[0].legend(handles=[
        mpatches.Patch(color="#5b8dd9", label="Baseline"),
        mpatches.Patch(color="#e05c5c", label="HRL-VGAE"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    save_to = out_path or os.path.join(ROOT_DIR, "hrl_vs_baselines.png")
    plt.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_to}")
    plt.close()


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
    ok = _run_pretrain_inline(args, TRAIN_DIR)
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
    ok = _run_pretrain_inline(args, train_dir)
    print("[Pretrain] Complete." if ok else "[Pretrain] Failed.", flush=True)


def _run_train(episodes, ll_pretrained, save_dir, train_dir,
               train_request_pct=DEFAULT_TRAIN_REQUEST_PCT):
    files = get_data_files(train_dir)
    if not files:
        print(f"[ERROR] No training files in {train_dir}.")
        return None

    _print_selected_files("TRAIN", files, request_pct=train_request_pct)

    strategy = None
    n_files = len(files)
    base_episodes = episodes // n_files
    extra = episodes % n_files

    for i, fp in enumerate(files):
        ep_for_file = base_episodes + (1 if i < extra else 0)
        if ep_for_file <= 0:
            continue

        print(f"\n--- File {i+1}/{n_files}: {os.path.basename(fp)} ({ep_for_file} ep) ---")
        env = load_env_from_json(fp, request_pct=train_request_pct)
        strategy = HRL_VGAE_Strategy(
            env, is_training=True, episodes=ep_for_file,
            use_ll_score=True,
            ll_pretrained_path=ll_pretrained if i == 0 else None)

        if i > 0:
            hl_w = os.path.join(save_dir, "hl_pmdrl_weights.npy")
            ll_w = os.path.join(save_dir, "ll_dqn_weights.npy")
            if os.path.exists(hl_w) or os.path.exists(ll_w):
                strategy.load_model(save_dir)

        env.set_strategy(strategy)
        env.run_simulation()
        os.makedirs(save_dir, exist_ok=True)
        strategy.save_model(save_dir)

    if strategy:
        strategy.save_model(save_dir)
    return strategy


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


def _run_eval(model_dir, test_dir, test_files=None,
              sample_n=None, sample_seed=None, csv_out=None):
    all_files = test_files or get_data_files(test_dir)
    if not all_files:
        print("[ERROR] No test files found.")
        return []

    files = _sample_files(all_files, sample_n, sample_seed)
    if sample_n and len(all_files) > len(files):
        print(f"[Eval] Sampled {len(files)}/{len(all_files)} files (seed={sample_seed})")

    results = []
    for fp in files:
        print(f"\n--- {os.path.basename(fp)} ---")
        env      = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
        if model_dir and os.path.isdir(model_dir):
            strategy.load_model(model_dir)
        env.set_strategy(strategy)
        stats = strategy.run_simulation_eval()
        env.print_statistics()
        results.append({
            "algorithm": "HRL-VGAE",
            "file":      os.path.basename(fp),
            "acceptance_ratio": round(stats.get("acceptance_ratio", 0), 4),
            "accepted":  stats.get("accepted_requests", 0),
            "rejected":  stats.get("rejected_requests", 0),
            "total_cost": round(stats.get("total_cost", 0), 2),
            "avg_cost":   round(stats.get("average_cost", 0), 2),
            "total_delay": round(stats.get("total_delay", 0), 2),
        })

    print("\n=== EVAL SUMMARY ===")
    print(f"{'File':<35} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<35} {r['acceptance_ratio']:>9.3f} {r['accepted']:>6} "
              f"{r['rejected']:>6} {r['total_cost']:>10.1f}")
    if results:
        avg_ar = sum(r["acceptance_ratio"] for r in results) / len(results)
        print(f"\nAverage acceptance ratio: {avg_ar:.3f}")

    out = csv_out or os.path.join(ROOT_DIR, "eval_results.csv")
    _save_csv(results, out)
    return results


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

    files = _sample_files(all_files, sample_n, sample_seed)
    if sample_n and len(all_files) > len(files):
        print(f"[Baseline] Sampled {len(files)}/{len(all_files)} files (seed={sample_seed})")

    print(f"\nBaseline comparison on {len(files)} file(s): {[os.path.basename(f) for f in files]}")

    # Per-file per-algorithm rows for CSV
    csv_rows = []
    # Aggregated for plotting
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

    # Build aggregated summary
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
    _save_csv(csv_rows, out, fieldnames=["algorithm","file","acceptance_ratio","accepted","rejected","total_cost","avg_cost","total_delay"])

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
        _save_csv(csv_rows, out, fieldnames=["algorithm","file","acceptance_ratio","accepted","rejected","total_cost","avg_cost","total_delay"])


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