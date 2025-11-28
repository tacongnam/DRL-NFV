import numpy as np

# --- CẤU HÌNH MẠNG & TÀI NGUYÊN ---
NUM_DCS = 6          # Số lượng Data Center
NUM_VNFS = 6         # Số loại VNF (NAT, FW, VOC, TM, WO, IDPS)
LINK_BANDWIDTH = 1024 # Mbps

# [Paper Section V.A]: DC Resources (CPU, RAM, Storage)
DC_RESOURCES_MIN = [12, 64, 500] 
DC_RESOURCES_MAX = [120, 256, 2000]

# --- CẤU HÌNH SFC (Table I) ---
# Mapping: 0:NAT, 1:FW, 2:VOC, 3:TM, 4:WO, 5:IDPS
# Format: { "Name": {"bw": Mbps, "delay": ms, "vnfs": [list_idx]} }
SFC_TYPES = {
    "CG":     {"bw": 4,     "delay": 80,  "vnfs": [0, 1, 2, 4, 5]},
    "AR":     {"bw": 100,   "delay": 10,  "vnfs": [0, 1, 3, 2, 5]},
    "VoIP":   {"bw": 0.064, "delay": 100, "vnfs": [0, 1, 3, 1, 0]},
    "VS":     {"bw": 4,     "delay": 100, "vnfs": [0, 1, 3, 2, 5]},
    "MIoT":   {"bw": 10,    "delay": 5,   "vnfs": [0, 1, 5]},
    "Ind4.0": {"bw": 70,    "delay": 8,   "vnfs": [0, 1]}
}

# Giả lập tài nguyên tiêu tốn cho mỗi loại VNF [CPU, RAM, Storage]
VNF_COSTS = [
    [2, 4, 10], [4, 8, 20], [2, 4, 10], 
    [1, 2, 5],  [2, 4, 10], [8, 16, 40]
]

# --- HYPERPARAMETERS ---
LATENT_DIM = 8       # Kích thước không gian ẩn Z của VAE
DQN_LR = 0.001
VAE_LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
GAMMA = 0.99