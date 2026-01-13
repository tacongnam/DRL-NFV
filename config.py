MAX_NUM_DCS = 50
LINK_BW_CAPACITY = 1000

MAX_VNF_TYPES = 10

ACTIONS_PER_TIME_STEP = 50
TIME_STEP = 1.0
MAX_SIM_TIME_PER_EPISODE = 100

TRAIN_EPISODES = 500
GENAI_DATA_EPISODES = 400
GENAI_VAE_EPOCHS = 150
GENAI_VALUE_EPOCHS = 300
GENAI_BATCH_SIZE = 256
GENAI_LATENT_DIM = 16
GENAI_MEMORY_SIZE = 100000
GENAI_SAMPLE_INTERVAL = 50

#DC_CPU_CYCLES = 12000
#DC_RAM = 256
#DC_STORAGE = 2048

# Thêm các biến toàn cục để lưu Max values thực tế
MAX_CPU = 1.0
MAX_RAM = 1.0
MAX_STORAGE = 1.0
MAX_BW = 1.0

def update_resource_constraints(dcs, graph):
    """Hàm này được gọi bởi Runner khi load dữ liệu mới"""
    global MAX_CPU, MAX_RAM, MAX_STORAGE, MAX_BW
    
    # Tìm Max Resource trong list DC
    servers = [d for d in dcs if d.is_server]
    if servers:
        MAX_CPU = max(max(d.cpu for d in servers), 1.0)
        MAX_RAM = max(max(d.ram for d in servers), 1.0)
        MAX_STORAGE = max(max(d.storage for d in servers), 1.0)
    
    # Tìm Max Bandwidth trong các cạnh của đồ thị
    max_bw = 0
    for u, v, data in graph.edges(data=True):
        # Lấy capacity gốc (thường lưu trong 'capacity' hoặc 'bw' lúc khởi tạo)
        c = data.get('capacity', data.get('bw', 0))
        if c > max_bw:
            max_bw = c
    MAX_BW = max(max_bw, 1.0)

    #print(f"  >>> Config Updated: MaxCPU={MAX_CPU:.1f}, MaxBW={MAX_BW:.1f}")

VNF_SPECS = {}
VNF_TYPES = []
NUM_VNF_TYPES = 0

def update_vnf_specs(vnf_specs_from_data):
    global VNF_SPECS, VNF_TYPES, NUM_VNF_TYPES
    VNF_SPECS = vnf_specs_from_data
    VNF_TYPES = list(VNF_SPECS.keys())
    NUM_VNF_TYPES = len(VNF_TYPES)

def get_action_space_size():
    return 2 * MAX_VNF_TYPES + 1

ACTION_SPACE_SIZE = 2 * MAX_VNF_TYPES + 1

LEARNING_RATE = 0.0005
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 100000

TARGET_NETWORK_UPDATE = 500
TRAIN_INTERVAL = 10

REWARD_SATISFIED = 5.0          # Tăng thưởng khi xong
REWARD_STEP_COMPLETED = 0.5     # Thêm: Thưởng khi đặt thành công 1 VNF (tiến bộ)
REWARD_DROPPED = -2.0
REWARD_INVALID = -1.0
REWARD_UNINSTALL_NEEDED = -0.5
REWARD_WAIT = 0.0

ALPHA_DELAY_PENALTY = 0.0
BETA_HOP_PENALTY = 0.0

# Paper values (from Algorithm 1)
PRIORITY_P2_SAME_DC = 10.0       # High priority for same DC (locality)
PRIORITY_P2_DIFF_DC = -10.0      # Penalty for cross-DC placement
URGENCY_THRESHOLD = 5.0          # Threshold for urgent requests (Thr in paper)
URGENCY_CONSTANT_C = 100.0       # Constant C for P3 calculation
EPSILON_SMALL = 1e-5             # Small epsilon to avoid division by zero

WEIGHTS_FILE = 'sfc_dqn.weights.h5'

TEST_EPSILON = 0.0
TEST_EPISODES = 1
TEST_NUM_DCS_RANGE = [2, 4, 6, 8]