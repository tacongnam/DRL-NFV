from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


class EpisodeMetrics:
    def __init__(self, time_horizon: float = None):
        self._revenues:  List[float] = []
        self._costs:     List[float] = []
        self._r2cs:      List[float] = []
        self._accepted:  int         = 0
        self._time_horizon           = time_horizon or 1.0

    def reset(self):
        self._revenues  = []
        self._costs     = []
        self._r2cs      = []
        self._accepted  = 0

    def record(self, revenue: float, cost: float, r2c: float, accepted: bool = True):
        if accepted:
            self._revenues.append(revenue)
            self._costs.append(cost)
            self._r2cs.append(r2c)
            self._accepted += 1

    def summary(self) -> Dict[str, float]:
        total_rev  = sum(self._revenues)
        total_cost = sum(self._costs)
        ltr        = total_rev / max(self._time_horizon, 1.0)
        lt_r2c     = (total_rev / max(total_cost, 1e-6)
                      if total_cost > 0 else 0.0)
        return {
            "ltr":        ltr,
            "lt_r2c":     lt_r2c,
            "total_rev":  total_rev,
            "total_cost": total_cost,
            "accepted":   self._accepted,
            "mean_r2c":   float(np.mean(self._r2cs)) if self._r2cs else 0.0,
        }


class ExperimentTracker:
    def __init__(self):
        self._algo_results: Dict[str, Dict] = {}

    def record_algo(self, algo_name: str, stats: dict):
        self._algo_results[algo_name] = stats

    def print_comparison_table(self):
        headers = ["Algorithm", "AR", "LTR", "LT-R2C", "Accepted", "Rejected"]
        fmt     = "{:<20} {:>8} {:>10} {:>10} {:>10} {:>10}"
        print("\n" + "=" * 70)
        print(fmt.format(*headers))
        print("-" * 70)
        for name, s in self._algo_results.items():
            ar    = s.get("acceptance_ratio", 0.0)
            ltr   = s.get("ltr", s.get("total_revenue", 0.0))
            r2c   = s.get("lt_r2c", 0.0)
            acc   = s.get("accepted_requests", 0)
            rej   = s.get("rejected_requests", 0)
            print(fmt.format(name, f"{ar:.3f}", f"{ltr:.2f}",
                             f"{r2c:.3f}", str(acc), str(rej)))
        print("=" * 70 + "\n")

    def print_admission_ablation(self):
        print("\n=== Admission Control Ablation ===")
        variants = ["DRL-NFV", "DRL-NoAdmission", "DRL-StaticThreshold"]
        for v in variants:
            if v in self._algo_results:
                s = self._algo_results[v]
                print(f"  {v:<25}  AR={s.get('acceptance_ratio', 0):.3f}"
                      f"  LTR={s.get('ltr', 0.0):.2f}"
                      f"  R2C={s.get('lt_r2c', 0.0):.3f}")

    def print_arrival_rate_table(self, results_by_rate: Dict[float, Dict[str, Dict]]):
        print("\n=== Arrival Rate Sensitivity ===")
        rates = sorted(results_by_rate.keys())
        algos = list(next(iter(results_by_rate.values())).keys()) if results_by_rate else []
        header = f"{'Rate':>6}  " + "  ".join(f"{a:<18}" for a in algos)
        print(header)
        print("-" * len(header))
        for rate in rates:
            row = f"{rate:>6.3f}  "
            for algo in algos:
                s  = results_by_rate[rate].get(algo, {})
                ar = s.get("acceptance_ratio", 0.0)
                row += f"AR={ar:.3f}           "
            print(row)