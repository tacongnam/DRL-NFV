import os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import load_env_from_json, BASELINE_REGISTRY
from strategy.hrl import HRL_VGAE_Strategy

# Cấu hình đường dẫn
MODEL_DIR = "models/hrl_final"
TEST_DIR = "data/test"
OUTPUT_DIR = "results_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_file_info(filename):
    """Phân tích thông tin từ tên file: topo_dist_diff.json"""
    parts = filename.replace(".json", "").split("_")
    return {
        "Topology": parts[0] if len(parts) > 0 else "Unknown",
        "Distribution": parts[1] if len(parts) > 1 else "Unknown",
        "Difficulty": parts[2] if len(parts) > 2 else "Unknown"
    }

def run_evaluation():
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".json")]
    all_results = []

    # Danh sách thuật toán bao gồm HRL-VGAE và Baselines
    algorithms = list(BASELINE_REGISTRY.keys()) + ["hrl-vgae"]

    print(f"Bắt đầu thực nghiệm trên {len(test_files)} kịch bản...")

    for fname in test_files:
        info = parse_file_info(fname)
        fpath = os.path.join(TEST_DIR, fname)
        
        for algo_key in algorithms:
            print(f"Đang chạy {algo_key} trên {fname}...")
            env = load_env_from_json(fpath)
            
            if algo_key == "hrl-vgae":
                strategy = HRL_VGAE_Strategy(env, is_training=False)
                if os.path.exists(MODEL_DIR):
                    strategy.load_model(MODEL_DIR)
                label = "HRL-VGAE"
            else:
                label, cls = BASELINE_REGISTRY[algo_key]
                strategy = cls(env)
            
            env.set_strategy(strategy)
            
            # Chạy mô phỏng (Dùng run_simulation_eval cho HRL để đảm bảo ko train)
            if algo_key == "hrl-vgae":
                stats = strategy.run_simulation_eval()
            else:
                stats = env.run_simulation()

            all_results.append({
                "Algorithm": label,
                "Topology": info["Topology"],
                "Distribution": info["Distribution"],
                "Difficulty": info["Difficulty"],
                "AR": stats.get("acceptance_ratio", 0),
                "Cost": stats.get("average_cost", 0),
                "Delay": stats.get("total_delay", 0) / max(stats.get("accepted_requests", 1), 1)
            })

    return pd.DataFrame(all_results)

def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # 1. Biểu đồ Acceptance Ratio theo Độ khó và Topo
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df, x="Difficulty", y="AR", hue="Algorithm", ax=ax1[0])
    ax1[0].set_title("Tỷ lệ chấp nhận (AR) theo Độ khó")
    
    sns.barplot(data=df, x="Topology", y="AR", hue="Algorithm", ax=ax1[1])
    ax1[1].set_title("Tỷ lệ chấp nhận (AR) theo Topology")
    plt.savefig(os.path.join(OUTPUT_DIR, "ar_comparison.png"))

    # 2. Biểu đồ Cost (Chi phí vận hành)
    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=df, x="Difficulty", y="Cost", hue="Algorithm", ax=ax2[0])
    ax2[0].set_title("Chi phí triển khai theo Độ khó")
    
    sns.barplot(data=df, x="Distribution", y="Cost", hue="Algorithm", ax=ax2[1])
    ax2[1].set_title("Chi phí triển khai theo Chiến lược phân bổ")
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_comparison.png"))

    # 3. BIỂU ĐỒ ĐA MỤC TIÊU (Scatter Pareto Plot)
    plt.figure(figsize=(10, 7))
    scatter = sns.scatterplot(data=df, x="Cost", y="AR", hue="Algorithm", style="Topology", s=100)
    plt.title("Phân tích Đa mục tiêu: AR vs Cost (Pareto Analysis)")
    plt.xlabel("Chi phí vận hành chuẩn hóa (Càng thấp càng tốt)")
    plt.ylabel("Tỷ lệ chấp nhận (Càng cao càng tốt)")
    
    # Vẽ mũi tên chỉ hướng tối ưu
    plt.annotate('Vùng tối ưu (Pareto)', xy=(df['Cost'].min(), df['AR'].max()), 
                 xytext=(df['Cost'].mean(), df['AR'].mean()),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10))
    
    plt.savefig(os.path.join(OUTPUT_DIR, "multi_objective_pareto.png"))
    plt.show()

if __name__ == "__main__":
    results_df = run_evaluation()
    
    # Lưu bảng kết quả
    summary = results_df.groupby(["Algorithm", "Difficulty", "Topology"]).mean().reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, "final_report.csv"), index=False)
    
    print("\n=== BẢNG KẾ QUẢ TỔNG HỢP ===")
    print(summary)
    
    # Vẽ biểu đồ
    plot_results(results_df)