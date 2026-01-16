MAX_NUM_DCS = 50
LINK_BW_CAPACITY = 1000

MAX_VNF_TYPES = 10

ACTIONS_PER_TIME_STEP = 25
TIME_STEP = 0.1
MAX_SIM_TIME_PER_EPISODE = 60

TRAIN_EPISODES = 200
GENAI_DATA_EPISODES = 100
GENAI_VAE_EPOCHS = 60
GENAI_VALUE_EPOCHS = 50
GENAI_BATCH_SIZE = 512
GENAI_LATENT_DIM = 16
GENAI_MEMORY_SIZE = 100000
GENAI_SAMPLE_INTERVAL = 5

MAX_CPU = 1.0
MAX_RAM = 1.0
MAX_STORAGE = 1.0
MAX_BW = 1.0

def update_resource_constraints(dcs, graph):
    global MAX_CPU, MAX_RAM, MAX_STORAGE, MAX_BW
    servers = [d for d in dcs if d.is_server]
    if servers:
        MAX_CPU = max(max(d.cpu for d in servers), 1.0)
        MAX_RAM = max(max(d.ram for d in servers), 1.0)
        MAX_STORAGE = max(max(d.storage for d in servers), 1.0)
    
    max_bw = 0
    for u, v, data in graph.edges(data=True):
        c = data.get('capacity', data.get('bw', 0))
        if c > max_bw: max_bw = c
    MAX_BW = max(max_bw, 1.0)

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

# --- CẢI TIẾN REWARD ---
# Tăng thưởng hoàn thành, giảm thưởng bước nhỏ, phạt nặng khi drop
REWARD_SATISFIED = 5.0          
REWARD_STEP_COMPLETED = 0.5
REWARD_DROPPED = -2.0
REWARD_INVALID = -1.0
REWARD_UNINSTALL_NEEDED = -0.1
REWARD_WAIT = 0.0

PRIORITY_P2_SAME_DC = 10.0
PRIORITY_P2_DIFF_DC = -10.0
URGENCY_THRESHOLD = 5.0
URGENCY_CONSTANT_C = 100.0
EPSILON_SMALL = 1e-5

WEIGHTS_FILE = 'sfc_dqn.weights.h5'

ALPHA_COST_PENALTY = 0.001