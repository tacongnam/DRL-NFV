import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_large_topology(json_path):
    # 1. Đọc dữ liệu
    with open(json_path, 'r') as f:
        data = json.load(f)

    G = nx.Graph()

    # 2. Phân loại tài nguyên để vẽ
    servers = []
    switches = []
    node_labels = {}
    
    for node_id, node_info in data['V'].items():
        is_server = node_info.get('server', False)
        G.add_node(node_id, is_server=is_server)
        
        if is_server:
            servers.append(node_id)
            # Chỉ hiện ID cho Server để tránh rối
            node_labels[node_id] = node_id 
        else:
            switches.append(node_id)

    for edge in data['E']:
        G.add_edge(str(edge['u']), str(edge['v']))

    # 3. Tính toán bố cục (Tăng tham số k để các node giãn xa nhau hơn)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # 4. Vẽ các thành phần theo lớp (Layering)
    
    # Vẽ Switches (Chấm nhỏ, không nhãn)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=switches, 
                           node_size=50, 
                           node_color='#5D1C6A', 
                           alpha=0.6, 
                           label='Switches')
    
    # Vẽ Servers (Chấm lớn hơn, màu nổi bật)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=servers, 
                           node_size=150, 
                           node_color='#CA5995', 
                           edgecolors='black', 
                           linewidths=1,
                           label='Servers (DC)')

    # Vẽ Cạnh (Link) - Dùng alpha thấp để làm mờ các đường kết nối
    nx.draw_networkx_edges(G, pos, width=1.1, edge_color='gray', alpha=0.3)

    # Vẽ nhãn - CHỈ cho Servers
    if node_labels:
        nx.draw_networkx_labels(G, pos, 
                                labels=node_labels, 
                                font_size=8, 
                                font_weight='bold',
                                verticalalignment='bottom')

    # 5. Hoàn thiện đồ thị
    plt.title(f"Topology Visualization: {len(data['V'])} Nodes", fontsize=16)
    plt.legend(scatterpoints=1, loc='upper left')
    plt.axis('off')
    
    # Lưu ảnh thay vì chỉ hiện (vì mạng lớn nhìn trên UI có thể lag)
    plt.savefig(f'{json_path}.png', dpi=300, bbox_inches='tight')
    print(f"Đã lưu sơ đồ mạng sạch vào file '{json_path}.png'")
    plt.show()

import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NFV VNF Placement – HRL-VGAE")
    p.add_argument("--file", default="test.json")
    # Đảm bảo file test.json nằm cùng thư mục hoặc thay đường dẫn tại đây
    args = p.parse_args()
    visualize_large_topology(args.file)