# config.py

# --- System Constants ---
MAX_NUM_DCS = 6    # Số lượng DC tối đa (để random từ 2 đến MAX)
LINK_BW_CAPACITY = 1000  # Mbps
SPEED_OF_LIGHT = 300000.0  # km/s

# --- Training & Simulation Hyperparameters (Paper Section III.A & IV) ---
TRAIN_UPDATES = 100          # U: Total updates
EPISODES_PER_UPDATE = 20     # E: Episodes per update
ACTIONS_PER_TIME_STEP = 100   # A: Actions per 1ms simulation step
TIME_STEP = 1       # T: 1 ms
TRAFFIC_GEN_INTERVAL = 4     # N: Generate traffic every 4 simulation steps (4ms)
MAX_SIM_TIME_PER_EPISODE = 200 # Giới hạn thời gian mô phỏng (ms) để tránh lặp vô tận
TRAFFIC_STOP_TIME = 150      # Dừng sinh request ở ms thứ 150 để hệ thống xử lý hết hàng đợi (End condition)

# --- DC Resources ---
DC_CPU_CYCLES = 120  # GHz
DC_RAM = 256         # GB
DC_STORAGE = 2048    # GB

# --- VNF Characteristics ---
VNF_SPECS = {
    'NAT':  {'cpu': 1, 'ram': 4,  'storage': 2,  'proc_time': 0.06},
    'FW':   {'cpu': 9, 'ram': 5,  'storage': 1,  'proc_time': 0.03},
    'VOC':  {'cpu': 5, 'ram': 11, 'storage': 13, 'proc_time': 0.11},
    'TM':   {'cpu': 13, 'ram': 7,  'storage': 7,  'proc_time': 0.07},
    'WO':   {'cpu': 5, 'ram': 2,  'storage': 5,  'proc_time': 0.08},
    'IDPS': {'cpu': 11, 'ram': 15, 'storage': 2,  'proc_time': 0.02},
}
VNF_TYPES = list(VNF_SPECS.keys())
NUM_VNF_TYPES = len(VNF_TYPES)

# --- SFC Characteristics ---
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

# --- DRL Parameters ---
ACTION_SPACE_SIZE = 2 * NUM_VNF_TYPES + 1
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99  # Giảm chậm hơn vì training lâu
EPSILON_MIN = 0.1
BATCH_SIZE = 64      # Tăng batch size cho ổn định
MEMORY_SIZE = 10000

# --- Reward Constants ---
REWARD_SATISFIED = 2.0
REWARD_DROPPED = -1.5
REWARD_INVALID = -1.0
REWARD_UNINSTALL_REQ = -0.5
REWARD_WAIT = 0.0

# --- Priority Calculation Constants (Algorithm 1) ---
# P2 Constants
PRIORITY_P2_SAME_DC = 50.0   # Thưởng lớn nếu VNF trước đó nằm cùng DC (High priority)
PRIORITY_P2_DIFF_DC = -20.0  # Phạt nếu VNF trước đó nằm khác DC (Degrade priority)

# P3 Constants
URGENCY_THRESHOLD = 20       # Ngưỡng 'Thr' (ms)
URGENCY_CONSTANT_C = 100.0   # Hằng số C
EPSILON = 1e-5               # Số rất nhỏ để tránh chia cho 0 (trong công thức P3)