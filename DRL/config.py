# config.py
# --- NETWORK CONFIGURATION ---
MAX_NUM_DCS = 6       
LINK_BW_CAPACITY = 1000 
SPEED_OF_LIGHT = 300000.0

# --- TRAINING HYPERPARAMETERS ---
TRAIN_UPDATES = 10             # U: Total updates (Paper says 350)
EPISODES_PER_UPDATE = 10       # E: Episodes per update (Paper says 20)
ACTIONS_PER_TIME_STEP = 100    # A: Actions per time step (Paper says 100)
TIME_STEP = 1                  # T: 1ms per step
TRAFFIC_GEN_INTERVAL = 4       # N: Generate every 4ms
TRAFFIC_STOP_TIME = 100        # Generate traffic until 100ms
MAX_SIM_TIME_PER_EPISODE = 100 

# --- RESOURCES ---
DC_CPU_CYCLES = 12000  # cycles/sec (Simplified relative to VNF req)
DC_RAM = 256           # GB
DC_STORAGE = 2048      # GB

# --- VNF & SFC SPECS (Based on Table I & II) ---
VNF_SPECS = {
    'NAT':  {'cpu': 1,  'ram': 4,  'storage': 2,  'proc_time': 0.06},
    'FW':   {'cpu': 9,  'ram': 5,  'storage': 1,  'proc_time': 0.03},
    'VOC':  {'cpu': 5,  'ram': 11, 'storage': 13, 'proc_time': 0.11},
    'TM':   {'cpu': 13, 'ram': 7,  'storage': 7,  'proc_time': 0.07},
    'WO':   {'cpu': 5,  'ram': 2,  'storage': 5,  'proc_time': 0.08},
    'IDPS': {'cpu': 11, 'ram': 15, 'storage': 2,  'proc_time': 0.02},
}
VNF_TYPES = list(VNF_SPECS.keys())
NUM_VNF_TYPES = len(VNF_TYPES)

SFC_SPECS = {
    'CloudGaming': {'chain': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'], 'bw': 4,   'delay': 80,  'bundle': (40, 55)},
    'AR':          {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 100, 'delay': 10,  'bundle': (1, 4)},
    'VoIP':        {'chain': ['NAT', 'FW', 'TM', 'FW', 'NAT'],   'bw': 0.064,'delay': 100, 'bundle': (100, 200)},
    'VideoStream': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 4,   'delay': 100, 'bundle': (50, 100)},
    'MIoT':        {'chain': ['NAT', 'FW', 'IDPS'],               'bw': 1,   'delay': 5,   'bundle': (10, 15)},
    'Ind4.0':      {'chain': ['NAT', 'FW'],                       'bw': 70,  'delay': 8,   'bundle': (1, 4)},
}
SFC_TYPES = list(SFC_SPECS.keys())
NUM_SFC_TYPES = len(SFC_TYPES)

# --- DRL SETTINGS ---
ACTION_SPACE_SIZE = 2 * NUM_VNF_TYPES + 1
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.9
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000

# --- REWARDS ---
REWARD_SATISFIED = 2.0
REWARD_DROPPED = -1.5
REWARD_INVALID = -1.0
REWARD_UNINSTALL_REQ = -0.5
REWARD_WAIT = 0.0 # Neutral

# --- PRIORITY CONSTANTS ---
PRIORITY_P2_SAME_DC = 10.0
PRIORITY_P2_DIFF_DC = -10.0
URGENCY_THRESHOLD = 5.0
URGENCY_CONSTANT_C = 100.0
EPSILON = 1e-5

WEIGHTS_FILE = 'sfc_dqn.weights.h5'

# Các tham số này cần thiết cho evaluate.py
TEST_EPSILON = 0.0            # Không explore khi test
TEST_EPISODES = 3            # Số episode chạy kiểm thử
TEST_FIG3_DCS = [2, 4, 6, 8]  # Các cấu hình số lượng DC để vẽ biểu đồ 3