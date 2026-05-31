"""
gen_tables.py
─────────────
Tạo bảng tổng hợp thực nghiệm theo 2 metrics (AR, EEI = AR/Cost),
tách riêng từng mức Difficulty (easy / normal / hard).

Chiều bảng:
  - Hàng  : Algorithm
  - Cột   : (Topology × Metric) và (Distribution × Metric)
  
Output mỗi difficulty:
  results_eval/table_{difficulty}.tex
  results_eval/table_{difficulty}.csv
"""

import os
import numpy as np
import pandas as pd

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CSV_PATH   = "baseline_results.csv"
OUTPUT_DIR = "results_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGO_ORDER = ["BestFit", "DeadlineAwareGreedy", "GreedyFIFS", "RandomFit", "DRL-NFV"]
DIFF_ORDER = ["easy", "normal", "hard"]
TOPO_ORDER = ["cogent", "conus", "nsf"]
DIST_ORDER = ["uniform", "urban", "rural", "centers"]

# Metrics cần tổng hợp: (tên cột trong df, nhãn LaTeX)
METRICS = [
    ("AR",  "AR"),
    ("EEI", "EEI"),
]


# ── Load & chuẩn bị dữ liệu ───────────────────────────────────────────────────

def parse_file_info(filename: str) -> dict:
    parts = str(filename).replace(".json", "").split("_")
    return {
        "Topology":     parts[0] if len(parts) > 0 else "Unknown",
        "Distribution": parts[1] if len(parts) > 1 else "Unknown",
        "Difficulty":   parts[2] if len(parts) > 2 else "Unknown",
    }


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    info = df["file"].apply(parse_file_info)
    df["Topology"]     = info.apply(lambda x: x["Topology"])
    df["Distribution"] = info.apply(lambda x: x["Distribution"])
    df["Difficulty"]   = info.apply(lambda x: x["Difficulty"])

    df = df.rename(columns={
        "algorithm":        "Algorithm",
        "acceptance_ratio": "AR",
        "avg_cost":         "Cost",
    })

    df["EEI"] = df["AR"] / df["Cost"].replace(0, np.nan)

    # Lọc chỉ giữ các giá trị hợp lệ
    df = df[df["Algorithm"].isin(ALGO_ORDER)]
    df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=ALGO_ORDER, ordered=True)

    return df


# ── Tạo bảng pivot ────────────────────────────────────────────────────────────

def build_pivot(df: pd.DataFrame, group_col: str, group_order: list) -> pd.DataFrame:
    """
    Tạo pivot: hàng = Algorithm, cột = (group_val, metric).
    Giá trị = mean qua tất cả các file thuộc nhóm đó.
    """
    agg = (
        df.groupby(["Algorithm", group_col], observed=True)[["AR", "EEI"]]
        .mean()
        .reset_index()
    )

    frames = []
    for grp in group_order:
        sub = agg[agg[group_col] == grp].set_index("Algorithm")[["AR", "EEI"]]
        sub.columns = pd.MultiIndex.from_tuples(
            [(grp, m) for m in ["AR", "EEI"]]
        )
        frames.append(sub)

    pivot = pd.concat(frames, axis=1).reindex(ALGO_ORDER)
    return pivot


# ── Đánh dấu best / second best ───────────────────────────────────────────────

def mark_best(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về DataFrame cùng shape với nhãn: 'best', 'second', '' cho mỗi ô.
    AR  → cao hơn tốt hơn
    EEI → cao hơn tốt hơn
    """
    marks = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for col in pivot.columns:
        vals = pivot[col].dropna().sort_values(ascending=False)
        if len(vals) >= 1:
            marks.loc[vals.index[0], col] = "best"
        if len(vals) >= 2:
            marks.loc[vals.index[1], col] = "second"
    return marks


# ── Render LaTeX ──────────────────────────────────────────────────────────────

def _fmt(val, mark: str) -> str:
    """Format một ô: 2 chữ số thập phân, bold nếu best, underline nếu second."""
    if pd.isna(val):
        return "--"
    s = f"{val:.4f}"
    if mark == "best":
        return r"\textbf{" + s + "}"
    if mark == "second":
        return r"\underline{" + s + "}"
    return s


def pivot_to_latex(
    topo_pivot: pd.DataFrame,
    dist_pivot: pd.DataFrame,
    topo_marks: pd.DataFrame,
    dist_marks: pd.DataFrame,
    difficulty: str,
) -> str:
    """Ghép 2 pivot (topology + distribution) thành 1 bảng LaTeX."""

    topo_cols = list(topo_pivot.columns)   # [(topo, metric), ...]
    dist_cols = list(dist_pivot.columns)   # [(dist, metric), ...]
    n_topo = len(topo_cols)
    n_dist = len(dist_cols)
    n_total = n_topo + n_dist

    # ── Column spec ──────────────────────────────────────────────────────────
    # l | ccc...ccc | ccc...ccc
    topo_spec = "".join(["c"] * n_topo)
    dist_spec = "".join(["c"] * n_dist)
    col_spec  = f"l|{topo_spec}|{dist_spec}"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    cap = (
        f"Kết quả thực nghiệm ở mức độ \\textbf{{{difficulty.capitalize()}}}. "
        r"Số in \textbf{đậm} = tốt nhất, \underline{gạch chân} = nhì. "
        r"EEI $= \mathrm{AR} / \mathrm{Cost}_{\mathrm{norm}}$."
    )
    lines.append(r"\caption{" + cap + "}")
    lines.append(r"\label{tab:results_" + difficulty + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # ── Header row 1: nhóm (Topology / Distribution) ─────────────────────────
    topo_groups = TOPO_ORDER
    dist_groups = DIST_ORDER

    h1_parts = [r"\multirow{2}{*}{\textbf{Algorithm}}"]
    for g in topo_groups:
        h1_parts.append(r"\multicolumn{2}{c}{" + g.capitalize() + "}")
    for g in dist_groups:
        h1_parts.append(r"\multicolumn{2}{c}{" + g.capitalize() + "}")
    lines.append(" & ".join(h1_parts) + r" \\")

    # Cmidrule dưới mỗi nhóm (bỏ qua cột Algorithm ở index 1)
    cmidrules = []
    col_idx = 2  # 1-based, cột 1 = Algorithm
    for _ in topo_groups:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    for _ in dist_groups:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
        col_idx += 2
    lines.append(" ".join(cmidrules))

    # ── Header row 2: AR / EEI cho mỗi nhóm ─────────────────────────────────
    h2_parts = [""]
    for _ in range(len(topo_groups) + len(dist_groups)):
        h2_parts.append("AR")
        h2_parts.append("EEI")
    lines.append(" & ".join(h2_parts) + r" \\")
    lines.append(r"\midrule")

    # ── Data rows ─────────────────────────────────────────────────────────────
    for algo in ALGO_ORDER:
        # Tên hiển thị — rút ngắn DeadlineAwareGreedy cho vừa cột
        display = algo.replace("DeadlineAwareGreedy", "DAGreedy")
        row_cells = [display]

        for col in topo_cols:
            val  = topo_pivot.loc[algo, col] if algo in topo_pivot.index else np.nan
            mark = topo_marks.loc[algo, col] if algo in topo_marks.index else ""
            row_cells.append(_fmt(val, mark))

        for col in dist_cols:
            val  = dist_pivot.loc[algo, col] if algo in dist_pivot.index else np.nan
            mark = dist_marks.loc[algo, col] if algo in dist_marks.index else ""
            row_cells.append(_fmt(val, mark))

        # Highlight DRL-NFV bằng màu nền nhẹ
        row_str = " & ".join(row_cells) + r" \\" 
        if algo == "DRL-NFV":
            row_str = r"\rowcolor{gray!12} " + row_str
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Render CSV (dễ import vào Excel hoặc paste thủ công) ─────────────────────

def pivots_to_csv(
    topo_pivot: pd.DataFrame,
    dist_pivot: pd.DataFrame,
) -> pd.DataFrame:
    """Ghép 2 pivot thành DataFrame phẳng, thêm tiền tố nhóm vào tên cột."""
    topo = topo_pivot.copy()
    topo.columns = [f"[Topo] {g} – {m}" for g, m in topo.columns]

    dist = dist_pivot.copy()
    dist.columns = [f"[Dist] {g} – {m}" for g, m in dist.columns]

    return pd.concat([topo, dist], axis=1).reset_index()


# ── Preamble gợi ý cho file LaTeX ─────────────────────────────────────────────

LATEX_PREAMBLE = r"""% Thêm vào preamble của file .tex chính:
% \usepackage{booktabs}
% \usepackage{multirow}
% \usepackage{colortbl}
% \usepackage{xcolor}
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Đọc dữ liệu: {CSV_PATH}")
    df = load_csv(CSV_PATH)
    print(f"  → {len(df)} dòng | Algorithms: {df['Algorithm'].unique().tolist()}\n")

    all_tex_blocks = [LATEX_PREAMBLE, "% ════════════════════════════════════\n"]

    for diff in DIFF_ORDER:
        sub = df[df["Difficulty"] == diff]
        if sub.empty:
            print(f"  ⚠ Không có dữ liệu cho difficulty='{diff}', bỏ qua.")
            continue

        print(f"Đang tạo bảng: {diff.upper()} ({len(sub)} dòng)...")

        topo_pivot = build_pivot(sub, "Topology",     TOPO_ORDER)
        dist_pivot = build_pivot(sub, "Distribution", DIST_ORDER)

        topo_marks = mark_best(topo_pivot)
        dist_marks = mark_best(dist_pivot)

        # ── LaTeX ──
        tex = pivot_to_latex(topo_pivot, dist_pivot,
                             topo_marks, dist_marks, diff)
        tex_path = os.path.join(OUTPUT_DIR, f"table_{diff}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"  ✓ LaTeX : {tex_path}")

        all_tex_blocks.append(f"% ── Difficulty: {diff.upper()} ──\n" + tex + "\n")

        # ── CSV ──
        csv_df = pivots_to_csv(topo_pivot, dist_pivot)
        csv_path = os.path.join(OUTPUT_DIR, f"table_{diff}.csv")
        csv_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  ✓ CSV   : {csv_path}")

    # Gộp tất cả bảng vào 1 file .tex tiện lợi
    combined_path = os.path.join(OUTPUT_DIR, "tables_all.tex")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_tex_blocks))
    print(f"\n✓ File tổng hợp: {combined_path}")
    print("  → Dùng \\input{{results_eval/tables_all.tex}} trong LaTeX chính.")


if __name__ == "__main__":
    main()