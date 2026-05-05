import os
from data.load_data import get_data_files, sample_files, load_env_from_json, save_csv

def _run_eval(model_dir, test_dir, test_files=None,
              sample_n=None, sample_seed=None, csv_out=None):
    all_files = test_files or get_data_files(test_dir)
    if not all_files:
        print("[ERROR] No test files found.")
        return []

    files = sample_files(all_files, sample_n, sample_seed)
    if sample_n and len(all_files) > len(files):
        print(f"[Eval] Sampled {len(files)}/{len(all_files)} files (seed={sample_seed})")

    results = []
    for fp in files:
        print(f"\n--- {os.path.basename(fp)} ---")
        env      = load_env_from_json(fp)
        from strategy.hrl import HRL_VGAE_Strategy
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

    out = csv_out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    save_csv(results, out)
    return results