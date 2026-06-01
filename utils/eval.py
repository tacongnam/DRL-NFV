import os
import time
import numpy as np
from data.load_data import get_data_files, sample_files, load_env_from_json, save_csv

def _run_eval(model_dir, test_dir, sample_n=None, sample_seed=None, num_runs=1):
    """
    Run evaluation on test dataset.
    
    Args:
        model_dir: Path to model directory with weights
        test_dir: Path to test dataset directory
        sample_n: Number of files to sample (None = all files)
        sample_seed: Random seed for sampling
        num_runs: Number of runs per test file (default=1)
    """
    all_files = get_data_files(test_dir)
    if not all_files:
        print("[ERROR] No test files found.")
        return []

    files = sample_files(all_files, sample_n, sample_seed)
    if sample_n and len(all_files) > len(files):
        print(f"[Eval] Sampled {len(files)}/{len(all_files)} files (seed={sample_seed})")
    
    if num_runs > 1:
        print(f"[Eval] Running {num_runs} trials per test file...")

    results = []
    for fp in files:
        filename = os.path.basename(fp)
        print(f"\n--- {filename} ---")
        
        run_results = []
        for run_id in range(num_runs):
            if num_runs > 1:
                print(f"  [Run {run_id + 1}/{num_runs}]", end=" ")
            
            t_start = time.time()
            env = load_env_from_json(fp)
            from strategy.drl_strategy import DRL_Strategy
            strategy = DRL_Strategy(env, is_training=False, episodes=1)
            if model_dir and os.path.isdir(model_dir):
                strategy.load_model(model_dir)
            env.set_strategy(strategy)
            stats = strategy.run_simulation_eval()
            if num_runs == 1:
                env.print_statistics()
            t_elapsed = time.time() - t_start
            
            run_result = {
                "acceptance_ratio": stats.get("acceptance_ratio", 0),
                "accepted": stats.get("accepted_requests", 0),
                "rejected": stats.get("rejected_requests", 0),
                "total_cost": stats.get("total_cost", 0),
                "avg_cost": stats.get("average_cost", 0),
                "total_delay": stats.get("total_delay", 0),
                "computing_time": t_elapsed,
            }
            run_results.append(run_result)
            
            if num_runs > 1:
                print(f"AR={run_result['acceptance_ratio']:.4f} Time={t_elapsed:.3f}s")
        
        # Calculate statistics across runs
        ar_values = [r["acceptance_ratio"] for r in run_results]
        cost_values = [r["total_cost"] for r in run_results]
        acc_cost_values = [r["avg_cost"] for r in run_results]
        delay_values = [r["total_delay"] for r in run_results]
        time_values = [r["computing_time"] for r in run_results]
        
        result = {
            "algorithm": "DRL-NFV",
            "file": filename,
            "num_runs": num_runs,
            "acceptance_ratio": round(np.mean(ar_values), 4),
            "acceptance_ratio_std": round(np.std(ar_values), 4) if num_runs > 1 else 0,
            "accepted": int(np.mean([r["accepted"] for r in run_results])),
            "rejected": int(np.mean([r["rejected"] for r in run_results])),
            "total_cost": round(np.mean(cost_values), 2),
            "total_cost_std": round(np.std(cost_values), 2) if num_runs > 1 else 0,
            "avg_cost": round(np.mean(acc_cost_values), 2),
            "avg_cost_std": round(np.std(acc_cost_values), 2) if num_runs > 1 else 0,
            "total_delay": round(np.mean(delay_values), 2),
            "total_delay_std": round(np.std(delay_values), 2) if num_runs > 1 else 0,
            "computing_time": round(np.mean(time_values), 3),
        }
        results.append(result)
        
        if num_runs > 1:
            print(f"  AVERAGE: AR={result['acceptance_ratio']:.4f}±{result['acceptance_ratio_std']:.4f} "
                  f"Cost={result['total_cost']:.2f}±{result['total_cost_std']:.2f} "
                  f"Time={result['computing_time']:.3f}s")
        else:
            print(f"[Eval] {filename:<40} acceptance ratio {result['acceptance_ratio']:.4f} completed in {result['computing_time']:.3f}s")

    # Print summary table
    print("\n=== EVAL SUMMARY ===")
    if num_runs > 1:
        print(f"{'File':<35} {'AccRatio':>12} {'Cost':>12} {'AvgCost':>12} {'Delay':>12} {'Time(s)':>8}")
        print("-"*105)
        for r in results:
            ar_str = f"{r['acceptance_ratio']:.4f}±{r['acceptance_ratio_std']:.4f}"
            cost_str = f"{r['total_cost']:.2f}±{r['total_cost_std']:.2f}"
            acc_cost_str = f"{r['avg_cost']:.2f}±{r['avg_cost_std']:.2f}"
            delay_str = f"{r['total_delay']:.2f}±{r['total_delay_std']:.2f}"
            print(f"{r['file']:<35} {ar_str:>12} {cost_str:>12} {acc_cost_str:>12} {delay_str:>12} {r['computing_time']:>8.3f}")
    else:
        print(f"{'File':<35} {'AccRatio':>9} {'Acc':>6} {'Rej':>6} {'Cost':>10} {'AvgCost':>10} {'Delay':>10} {'Time(s)':>8}")
        print("-"*110)
        for r in results:
            print(f"{r['file']:<35} {r['acceptance_ratio']:>9.3f} {r['accepted']:>6} "
                  f"{r['rejected']:>6} {r['total_cost']:>10.1f} {r['avg_cost']:>10.2f} {r['total_delay']:>10.1f} {r['computing_time']:>8.3f}")
    
    # Print average metrics
    if results:
        avg_ar = sum(r["acceptance_ratio"] for r in results) / len(results)
        avg_cost = sum(r["total_cost"] for r in results) / len(results)
        avg_delay = sum(r["total_delay"] for r in results) / len(results)
        avg_acc_cost = sum(r["avg_cost"] for r in results) / len(results)
        avg_time = sum(r["computing_time"] for r in results) / len(results)
        print(f"\n{'=== AVERAGE METRICS ===':<35}")
        print(f"  Acceptance Ratio: {avg_ar:.4f}")
        print(f"  Total Cost:       {avg_cost:.2f}")
        print(f"  Avg Cost/Request: {avg_acc_cost:.2f}")
        print(f"  Total Delay:      {avg_delay:.2f}")
        print(f"  Avg Time (s):     {avg_time:.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    save_csv(results, out)
    return results