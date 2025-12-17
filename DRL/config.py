# config.py
# --- NETWORK CONFIGURATION ---
MAX_NUM_DCS = 6       
LINK_BW_CAPACITY = 1000  # Mbps
SPEED_OF_LIGHT = 300000.0  # km/s

# --- TRAINING HYPERPARAMETERS ---
TRAIN_UPDATES = 30             # U: Total updates
EPISODES_PER_UPDATE = 10       # E: Episodes per update
ACTIONS_PER_TIME_STEP = 50     # A: Actions per time step
TIME_STEP = 1                  # T: 1ms per step
TRAFFIC_GEN_INTERVAL = 5       # N: Generate every 5ms
TRAFFIC_STOP_TIME = 50         # Generate traffic until 50ms
MAX_SIM_TIME_PER_EPISODE = 150 # Maximum simulation time

# --- GENAI SPECIFIC PARAMS ---
GENAI_DATA_EPISODES = 20       # Reduced from 100
GENAI_VAE_EPOCHS = 20          # Reduced from 50
GENAI_VALUE_EPOCHS = 10        # Reduced from 30
GENAI_BATCH_SIZE = 64          # Increased for efficiency
GENAI_LATENT_DIM = 16          # Reduced from 32
GENAI_MEMORY_SIZE = 20000      # Reduced from 50000
GENAI_SAMPLE_INTERVAL = 200    # Collect every 200 steps instead of 100

# --- RESOURCES ---
DC_CPU_CYCLES = 12000  # cycles/sec
DC_RAM = 256           # GB
DC_STORAGE = 2048      # GB

# --- VNF & SFC SPECS ---
VNF_SPECS = {
    'NAT':  {'cpu': 1,  'ram': 4,  'storage': 7,  'proc_time': 0.12},
    'FW':   {'cpu': 9,  'ram': 5,  'storage': 1,  'proc_time': 0.06},
    'VOC':  {'cpu': 5,  'ram': 11, 'storage': 13, 'proc_time': 0.22},
    'TM':   {'cpu': 13, 'ram': 7,  'storage': 7,  'proc_time': 0.14},
    'WO':   {'cpu': 5,  'ram': 2,  'storage': 5,  'proc_time': 0.16},
    'IDPS': {'cpu': 11, 'ram': 15, 'storage': 2,  'proc_time': 0.04},
}
VNF_TYPES = list(VNF_SPECS.keys())  
NUM_VNF_TYPES = len(VNF_TYPES)

SFC_SPECS = {
    'CloudGaming': {'chain': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'], 'bw': 4,   'delay': 80,  'bundle': (20, 22)},
    'AR':          {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 100, 'delay': 10,  'bundle': (1, 4)},
    'VoIP':        {'chain': ['NAT', 'FW', 'TM', 'FW', 'NAT'],   'bw': 0.064,'delay': 100, 'bundle': (50, 100)},
    'VideoStream': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 4,   'delay': 100, 'bundle': (25, 50)},
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
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000

TARGET_NETWORK_UPDATE = 50000
TRAIN_INTERVAL = 200

# --- REWARDS ---
REWARD_SATISFIED = 2.0
REWARD_DROPPED = -1.5
REWARD_INVALID = -1.0
REWARD_UNINSTALL_NEEDED = -0.5
REWARD_WAIT = 0.0

# --- PRIORITY CONSTANTS ---
PRIORITY_P2_SAME_DC = 10.0
PRIORITY_P2_DIFF_DC = -10.0
URGENCY_THRESHOLD = 5.0
URGENCY_CONSTANT_C = 100.0
EPSILON_SMALL = 1e-5

WEIGHTS_FILE = 'sfc_dqn.weights.h5'

# Test parameters
TEST_EPSILON = 0.0
TEST_EPISODES = 3  # Reduced from 10
TEST_FIG3_DCS = [2, 4, 6, 8]