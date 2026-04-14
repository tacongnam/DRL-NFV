import os, sys, json, argparse, subprocess, math
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
PRETRAIN_SCRIPT  = os.path.join(ROOT_DIR, "models/pretrain.py")
DEFAULT_EPISODES = 60
DEFAULT_TRAIN_PROFILE = "fast"
DEFAULT_MAX_TRAIN_FILES = 3
DEFAULT_MAX_PRETRAIN_FILES = 3
DEFAULT_MAX_TRAIN_REQUESTS = 120
DEFAULT_MAX_PRETRAIN_REQUESTS = 160
DEFAULT_MIN_SERVERS = 3
DEFAULT_MIN_SERVER_RATIO = 0.12
DEFAULT_REQUEST_SAMPLE_MODE = "uniform"
DEFAULT_TRAIN_REQUEST_PCT = 10
DEFAULT_PRETRAIN_REQUEST_PCT = 15

BASELINE_REGISTRY = {
    "fifs":     ("GreedyFIFS",          GreedyFIFS),
    "glb":      ("GreedyGLB",           GreedyGLB),
    "spf":      ("ShortestPathFirst",   ShortestPathFirst),
    "bestfit":  ("BestFit",             BestFit),
    "deadline": ("DeadlineAwareGreedy", DeadlineAwareGreedy),
    "random":   ("RandomFit",           RandomFit),
}


def _sample_requests(req_rows: list, max_requests: int = None,
                     sample_mode: str = DEFAULT_REQUEST_SAMPLE_MODE) -> list:
    if max_requests is None or max_requests <= 0 or len(req_rows) <= max_requests:
        return req_rows
    if sample_mode == "head":
        return req_rows[:max_requests]
    idxs = np.linspace(0, len(req_rows) - 1, num=max_requests, dtype=int)
    return [req_rows[i] for i in idxs]


def _resolve_request_limit(total_requests: int, max_requests: int = None,
                           request_pct: int = None) -> int | None:
    limits = []
    if max_requests is not None and max_requests > 0:
        limits.append(max_requests)
    if request_pct is not None and request_pct > 0:
        limits.append(max(1, math.ceil(total_requests * request_pct / 100.0)))
    return min(limits) if limits else None


def load_env_from_json(filepath: str, max_requests: int = None,
                       sample_mode: str = DEFAULT_REQUEST_SAMPLE_MODE,
                       request_pct: int = None) -> Env:
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
    req_limit = _resolve_request_limit(len(req_rows), max_requests=max_requests, request_pct=request_pct)
    req_rows = _sample_requests(req_rows, max_requests=req_limit, sample_mode=sample_mode)

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


def _read_dataset_meta(filepath: str) -> dict:
    with open(filepath) as f:
        data = json.load(f)
    requests = data.get("R", [])
    total_requests = len(requests)
    server_count = sum(1 for node in data.get("V", {}).values() if node.get("server", False))
    node_count = len(data.get("V", {}))
    avg_bw = (sum(r.get("b_r", 0.0) for r in requests) / total_requests) if total_requests else 0.0
    avg_delay = (sum(r.get("d_max", 0.0) for r in requests) / total_requests) if total_requests else 0.0
    parts = os.path.basename(filepath).replace(".json", "").split("_")
    topology = parts[0] if parts else "unknown"
    distribution = parts[1] if len(parts) > 1 else "unknown"
    difficulty = parts[2] if len(parts) > 2 else "unknown"
    return {
        "path": filepath,
        "name": os.path.basename(filepath),
        "topology": topology,
        "distribution": distribution,
        "difficulty": difficulty,
        "nodes": node_count,
        "servers": server_count,
        "server_ratio": (server_count / node_count) if node_count else 0.0,
        "requests": total_requests,
        "avg_bw": avg_bw,
        "avg_delay": avg_delay,
    }


def _select_dataset_files(files: list, profile: str = DEFAULT_TRAIN_PROFILE,
                          max_files: int = None,
                          min_servers: int = DEFAULT_MIN_SERVERS,
                          min_server_ratio: float = DEFAULT_MIN_SERVER_RATIO) -> list:
    if profile == "full":
        selected = [_read_dataset_meta(fp) for fp in files]
    else:
        metas = [_read_dataset_meta(fp) for fp in files]
        filtered = [
            m for m in metas
            if m["difficulty"] == "easy"
            and m["servers"] >= min_servers
            and m["server_ratio"] >= min_server_ratio
        ]
        if not filtered:
            filtered = metas
        selected = sorted(
            filtered,
            key=lambda m: (
                m["nodes"],
                -m["server_ratio"],
                m["avg_bw"],
                -m["avg_delay"],
                m["name"],
            ),
        )
        if profile == "balanced":
            selected = selected[: max_files or len(selected)]
    if max_files and max_files > 0:
        selected = selected[:max_files]
    return selected


def _print_selected_files(label: str, selected: list, max_requests: int = None,
                          request_pct: int = None):
    print(f"\n[{label}] Selected {len(selected)} file(s)")
    for meta in selected:
        req_limit = _resolve_request_limit(meta["requests"], max_requests=max_requests, request_pct=request_pct)
        req_label = meta["requests"] if req_limit is None else min(meta["requests"], req_limit)
        print(
            f"  - {meta['name']}: nodes={meta['nodes']} servers={meta['servers']}"
            f" req={req_label}/{meta['requests']} bw={meta['avg_bw']:.2f}"
            f" delay={meta['avg_delay']:.2f}"
        )


def _add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--train-dir", default=TRAIN_DIR)
    parser.add_argument("--model-dir", default="models/hrl_final")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--test-files", nargs="+", default=None)
    parser.add_argument("--ll-pretrained", type=str, default=None)


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
    parser.add_argument("--train-profile", type=str, default=DEFAULT_TRAIN_PROFILE,
                        choices=["fast", "balanced", "full"])
    parser.add_argument("--max-train-files", type=int, default=DEFAULT_MAX_TRAIN_FILES)
    parser.add_argument("--max-pretrain-files", type=int, default=DEFAULT_MAX_PRETRAIN_FILES)
    parser.add_argument("--max-train-requests", type=int, default=DEFAULT_MAX_TRAIN_REQUESTS)
    parser.add_argument("--max-pretrain-requests", type=int, default=DEFAULT_MAX_PRETRAIN_REQUESTS)
    parser.add_argument("--train-request-pct", type=int, default=DEFAULT_TRAIN_REQUEST_PCT)
    parser.add_argument("--pretrain-request-pct", type=int, default=DEFAULT_PRETRAIN_REQUEST_PCT)
    parser.add_argument("--request-sample-mode", type=str, default=DEFAULT_REQUEST_SAMPLE_MODE,
                        choices=["uniform", "head"])
    parser.add_argument("--min-servers", type=int, default=DEFAULT_MIN_SERVERS)
    parser.add_argument("--min-server-ratio", type=float, default=DEFAULT_MIN_SERVER_RATIO)


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

    selected = pretrain_module.select_train_files(
        train_dir,
        profile=getattr(args, "train_profile", DEFAULT_TRAIN_PROFILE),
        max_files=getattr(args, "max_pretrain_files", DEFAULT_MAX_PRETRAIN_FILES),
        min_servers=getattr(args, "min_servers", DEFAULT_MIN_SERVERS),
        min_server_ratio=getattr(args, "min_server_ratio", DEFAULT_MIN_SERVER_RATIO),
    )
    if not selected:
        print("[Pretrain] No training files selected.", flush=True)
        return False

    req_pct = getattr(args, "pretrain_request_pct", DEFAULT_PRETRAIN_REQUEST_PCT)
    req_max = getattr(args, "max_pretrain_requests", DEFAULT_MAX_PRETRAIN_REQUESTS)
    pretrain_module.print_selected_files(selected, req_max, req_pct)

    print(f"[Pretrain] Running inline on {train_dir}", flush=True)
    vgae = None
    vgae = pretrain_module.pretrain_vgae(
        selected,
        epochs=getattr(args, "vgae_epochs", 60),
        max_requests=req_max,
        sample_mode=getattr(args, "request_sample_mode", DEFAULT_REQUEST_SAMPLE_MODE),
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
            max_requests=req_max,
            sample_mode=getattr(args, "request_sample_mode", DEFAULT_REQUEST_SAMPLE_MODE),
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
    ll_path = os.path.join(ROOT_DIR, "models/ll_pretrained/ll_dqn_weights.weights.h5")
    _run_train(
        args.episodes,
        ll_path if os.path.exists(ll_path) else None,
        os.path.join(ROOT_DIR, "models/hrl_final"),
        TRAIN_DIR,
        train_profile=getattr(args, "train_profile", DEFAULT_TRAIN_PROFILE),
        max_train_files=getattr(args, "max_train_files", DEFAULT_MAX_TRAIN_FILES),
        max_train_requests=getattr(args, "max_train_requests", DEFAULT_MAX_TRAIN_REQUESTS),
        train_request_pct=getattr(args, "train_request_pct", DEFAULT_TRAIN_REQUEST_PCT),
        request_sample_mode=getattr(args, "request_sample_mode", DEFAULT_REQUEST_SAMPLE_MODE),
        min_servers=getattr(args, "min_servers", DEFAULT_MIN_SERVERS),
        min_server_ratio=getattr(args, "min_server_ratio", DEFAULT_MIN_SERVER_RATIO),
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
               train_profile=DEFAULT_TRAIN_PROFILE,
               max_train_files=DEFAULT_MAX_TRAIN_FILES,
               max_train_requests=DEFAULT_MAX_TRAIN_REQUESTS,
               train_request_pct=DEFAULT_TRAIN_REQUEST_PCT,
               request_sample_mode=DEFAULT_REQUEST_SAMPLE_MODE,
               min_servers=DEFAULT_MIN_SERVERS,
               min_server_ratio=DEFAULT_MIN_SERVER_RATIO):
    files = get_data_files(train_dir)
    if not files:
        print(f"[ERROR] No training files in {train_dir}.")
        return None

    selected = _select_dataset_files(
        files,
        profile=train_profile,
        max_files=max_train_files,
        min_servers=min_servers,
        min_server_ratio=min_server_ratio,
    )
    if not selected:
        print("[ERROR] No suitable training files selected.")
        return None

    _print_selected_files("TRAIN", selected, max_requests=max_train_requests, request_pct=train_request_pct)

    strategy = None
    n_files = len(selected)
    base_episodes = episodes // n_files
    extra = episodes % n_files

    for i, meta in enumerate(selected):
        ep_for_file = base_episodes + (1 if i < extra else 0)
        if ep_for_file <= 0:
            continue

        fp = meta["path"]
        print(f"\n--- File {i+1}/{n_files}: {meta['name']} ({ep_for_file} ep) ---")
        env = load_env_from_json(
            fp,
            max_requests=max_train_requests,
            sample_mode=request_sample_mode,
            request_pct=train_request_pct,
        )
        strategy = HRL_VGAE_Strategy(
            env, is_training=True, episodes=ep_for_file,
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
    if not ll_path:
        candidate = os.path.join(ROOT_DIR, "models/ll_pretrained/ll_dqn_weights.weights.h5")
        ll_path   = candidate if os.path.exists(candidate) else None
    _run_train(args.episodes, ll_path,
               os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
               os.path.abspath(getattr(args, "train_dir", TRAIN_DIR)),
               train_profile=getattr(args, "train_profile", DEFAULT_TRAIN_PROFILE),
               max_train_files=getattr(args, "max_train_files", DEFAULT_MAX_TRAIN_FILES),
               max_train_requests=getattr(args, "max_train_requests", DEFAULT_MAX_TRAIN_REQUESTS),
               train_request_pct=getattr(args, "train_request_pct", DEFAULT_TRAIN_REQUEST_PCT),
               request_sample_mode=getattr(args, "request_sample_mode", DEFAULT_REQUEST_SAMPLE_MODE),
               min_servers=getattr(args, "min_servers", DEFAULT_MIN_SERVERS),
               min_server_ratio=getattr(args, "min_server_ratio", DEFAULT_MIN_SERVER_RATIO))


def _run_eval(model_dir, test_dir, test_files=None):
    files = test_files or get_data_files(test_dir)
    if not files:
        print("[ERROR] No test files found.")
        return []

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
            "name":  "HRL-VGAE",
            "file":  os.path.basename(fp),
            "ar":    stats.get("acceptance_ratio", 0),
            "acc":   stats.get("accepted_requests", 0),
            "rej":   stats.get("rejected_requests", 0),
            "cost":  stats.get("total_cost", 0),
            "delay": stats.get("total_delay", 0),
        })

    print("\n=== EVAL SUMMARY ===")
    print(f"{'File':<35} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<35} {r['ar']:>9.3f} {r['acc']:>6} {r['rej']:>6} {r['cost']:>10.1f}")
    if results:
        print(f"\nAverage acceptance ratio: {sum(r['ar'] for r in results)/len(results):.3f}")
    return results


def run_eval(args):
    print("\n=== EVALUATION ===")
    _run_eval(os.path.abspath(getattr(args, "model_dir", "models/hrl_final")),
              os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR),
              getattr(args, "test_files", None))


def run_baselines(args=None):
    baselines_to_run = getattr(args, "baselines", None) or list(BASELINE_REGISTRY.keys())
    plot_out         = getattr(args, "plot_out", None)
    test_dir         = os.path.abspath(getattr(args, "test_dir", None) or TEST_DIR)

    files = get_data_files(test_dir) or get_data_files(os.path.join(ROOT_DIR, "data"))
    if not files:
        print("[ERROR] No test files found. Run --mode generate first.")
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
        env.set_strategy(cls(env))
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
        print("-"*75)
        for r in results:
            print(f"{r['name']:<25} {r['ar']:>9.3f} {r['acc']:>6} {r['rej']:>6} "
                  f"{r['cost']:>10.1f} {r['delay']:>10.1f}")
        _plot_baseline_results(results, out_path=plot_out)

    model_dir = os.path.abspath(getattr(args, "model_dir", "models/hrl_final"))
    hrl_weights = ["hl_pmdrl_weights.weights.h5", "ll_dqn_weights.weights.h5"]
    if os.path.isdir(model_dir) and any(
        os.path.exists(os.path.join(model_dir, w)) for w in hrl_weights
    ):
        print(f"\n[HRL-VGAE] Evaluating from {model_dir} ...")
        env      = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(env, is_training=False, episodes=1)
        strategy.load_model(model_dir)
        env.set_strategy(strategy)
        hrl_stats = strategy.run_simulation_eval()
        env.print_statistics()
        hrl_r = [{
            "name":  "HRL-VGAE",
            "ar":    hrl_stats.get("acceptance_ratio", 0.0),
            "cost":  hrl_stats.get("total_cost", 0.0),
            "delay": hrl_stats.get("total_delay", 0.0),
        }]
        cmp_out = (plot_out.replace(".png", "_vs_hrl.png") if plot_out
                   else os.path.join(ROOT_DIR, "hrl_vs_baselines.png"))
        _plot_eval_vs_baselines(hrl_r, results, out_path=cmp_out)


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
