import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Cấu hình phông chữ đẹp hơn
plt.rcParams.update({'font.size': 11})

def load_results(json_file):
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return []
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['files']

def parse_group(filename):
    """Phân loại file vào nhóm Topology và Difficulty"""
    name = filename.lower()
    
    # Topology
    if 'cogent' in name: topo = 'COGENT'
    elif 'conus' in name: topo = 'CONUS'
    elif 'nsf' in name: topo = 'NSF'
    else: topo = 'OTHER'
    
    # Difficulty
    if 'hard' in name: diff = 'Hard'
    elif 'easy' in name: diff = 'Easy'
    else: diff = 'Normal'
    
    return topo, diff

def plot_paper_style_grouped(data):
    """
    Vẽ biểu đồ Scatter giống bài báo nhưng nhóm theo Topology để dễ hiểu hơn.
    """
    # 1. Sắp xếp dữ liệu theo nhóm: NSF -> CONUS -> COGENT (từ nhỏ đến lớn)
    # Và trong mỗi nhóm sắp xếp theo độ khó: Easy -> Normal -> Hard
    
    def sort_key(item):
        topo, diff = parse_group(item['file'])
        topo_order = {'NSF': 0, 'CONUS': 1, 'COGENT': 2, 'OTHER': 3}
        diff_order = {'Easy': 0, 'Normal': 1, 'Hard': 2}
        return (topo_order.get(topo, 3), diff_order.get(diff, 1))

    data.sort(key=sort_key)

    files = [d['file'] for d in data]
    dqn_cost = [d['dqn']['cost'] for d in data]
    vae_cost = [d['vae']['cost'] for d in data]
    
    # Chuẩn hóa Delay về 0-1 để vẽ (Lấy max delay của cả tập dữ liệu làm mẫu số)
    raw_dqn_delay = [d['dqn']['delay'] for d in data]
    raw_vae_delay = [d['vae']['delay'] for d in data]
    max_delay = max(max(raw_dqn_delay), max(raw_vae_delay))
    
    dqn_delay = [d / max_delay for d in raw_dqn_delay]
    vae_delay = [d / max_delay for d in raw_vae_delay]

    # Tạo X-axis indices
    x = np.arange(len(data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- SUBPLOT 1: DELAY ---
    ax1.scatter(x, dqn_delay, color='#1f77b4', label='DQN', alpha=0.7, s=150, marker='o')
    ax1.scatter(x, vae_delay, color='#ff7f0e', label='VAE-DQN', alpha=0.9, s=150, marker='^')
    
    # Vẽ các đường phân chia Topology
    prev_topo = None
    ticks = []
    tick_labels = []
    
    for i, item in enumerate(data):
        topo, _ = parse_group(item['file'])
        if topo != prev_topo:
            if i > 0:
                ax1.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
                ax2.axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            prev_topo = topo
            # Lưu vị trí để ghi nhãn trục X
            ticks.append(i)
            tick_labels.append(topo)
    
    # Căn chỉnh nhãn trục X vào giữa các vùng
    final_ticks = []
    for j in range(len(ticks)):
        start = ticks[j]
        end = ticks[j+1] if j+1 < len(ticks) else len(data)
        final_ticks.append((start + end - 1) / 2)

    ax1.set_xticks(final_ticks)
    ax1.set_xticklabels(tick_labels, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Delay Time', fontsize=12)
    ax1.set_title('(a) Delay Time Comparison', y=-0.15, fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- SUBPLOT 2: COST ---
    ax2.scatter(x, dqn_cost, color='#1f77b4', label='DQN', alpha=0.7, s=150, marker='o')
    ax2.scatter(x, vae_cost, color='#ff7f0e', label='VAE-DQN', alpha=0.9, s=150, marker='^')
    
    ax2.set_xticks(final_ticks)
    ax2.set_xticklabels(tick_labels, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Deployment Cost', fontsize=12)
    ax2.set_title('(b) Network Deployment Cost', y=-0.15, fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Comparison: DQN vs VAE-DQN (Grouped by Topology)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    os.makedirs("fig", exist_ok=True)
    save_path = "fig/final_comparison_scatter.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Scatter plot saved to: {save_path}")

def plot_hard_scenario_bar(data):
    """
    Vẽ biểu đồ cột CHỈ tập trung vào các trường hợp HARD (nơi VAE tỏa sáng).
    So sánh Acceptance Ratio (AR) và Cost.
    """
    hard_data = [d for d in data if 'hard' in d['file'].lower()]
    
    if not hard_data:
        print("No hard scenarios found to plot.")
        return

    labels = []
    dqn_ar = []
    vae_ar = []
    
    # Lấy tên ngắn gọn
    for d in hard_data:
        topo, _ = parse_group(d['file'])
        # Tên hiển thị: COGENT-Hard, NSF-Hard...
        name = d['file'].replace('_hard_s1.json', '').replace('_s1.json', '')
        name = name.replace('cogent', 'CG').replace('conus', 'CN').replace('nsf', 'NSF')
        name = name.replace('_centers', '-C').replace('_uniform', '-U').replace('_rural', '-R').replace('_urban', '-Ub')
        labels.append(name)
        dqn_ar.append(d['dqn']['ar'])
        vae_ar.append(d['vae']['ar'])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, dqn_ar, width, label='DQN', color='#d62728', alpha=0.8)
    rects2 = ax.bar(x + width/2, vae_ar, width, label='VAE-DQN', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('Acceptance Ratio (%)')
    ax.set_title('Performance in "Hard" Scenarios (Network Saturation)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Thêm text label trên cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    save_path = "fig/final_hard_scenarios_bar.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Bar chart saved to: {save_path}")

if __name__ == "__main__":
    json_path = 'comparison_results.json'
    results = load_results(json_path)
    
    if results:
        plot_paper_style_grouped(results)
        plot_hard_scenario_bar(results)
        print("\nDone! Check the 'fig/' folder.")
    else:
        print("No data found. Please run 'python main.py compare' first.")