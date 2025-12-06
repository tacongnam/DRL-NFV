MAX_NUM_DCS = 6    # Maximum number of data centers (random 2-6 during training)
LINK_BW_CAPACITY = 1000  # Mbps
SPEED_OF_LIGHT = 300000.0  # km/s

TRAIN_UPDATES = 40          # U: Total weight updates
EPISODES_PER_UPDATE = 20     # E: Episodes collected before each update
ACTIONS_PER_TIME_STEP = 100   # A: Number of actions per 1ms timestep
                              # Paper uses 100, but code uses 50 (faster training)
TIME_STEP = 1                 # T: Duration of each timestep in milliseconds
                              # Action inference time = T / A = 1ms / 50 = 0.02ms

TRAFFIC_GEN_INTERVAL = 4     # N: Generate new SFC requests every 4 timesteps (4ms)
MAX_SIM_TIME_PER_EPISODE = 150 # Maximum simulation time per episode (ms)
TRAFFIC_STOP_TIME = 150      # Stop generating new traffic at 150ms
                              # Episode continues until all active requests are processed

DC_CPU_CYCLES = 120  # GHz - Computational capacity per DC
DC_RAM = 256         # GB - Memory capacity per DC
DC_STORAGE = 2048    # GB - Storage capacity per DC

VNF_SPECS = {
    # Format: {'cpu': GHz, 'ram': GB, 'storage': GB, 'proc_time': ms}
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
    # Format: {'chain': [VNF sequence], 'bw': Mbps, 'delay': ms, 'bundle': (min, max)}
    'CloudGaming': {'chain': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'], 
                    'bw': 4,   'delay': 80,  'bundle': (40, 55)},
    'AR':          {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 
                    'bw': 100, 'delay': 10,  'bundle': (1, 4)},
    'VoIP':        {'chain': ['NAT', 'FW', 'TM', 'FW', 'NAT'],   
                    'bw': 0.064, 'delay': 100, 'bundle': (100, 200)},
    'VideoStream': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 
                    'bw': 4,   'delay': 100, 'bundle': (50, 100)},
    'MIoT':        {'chain': ['NAT', 'FW', 'IDPS'],               
                    'bw': 1,   'delay': 5,   'bundle': (10, 15)},
    'Ind4.0':      {'chain': ['NAT', 'FW'],                       
                    'bw': 70,  'delay': 8,   'bundle': (1, 4)},
}
SFC_TYPES = list(SFC_SPECS.keys())
NUM_SFC_TYPES = len(SFC_TYPES)

ACTION_SPACE_SIZE = 2 * NUM_VNF_TYPES + 1  # WAIT + UNINSTALL×6 + ALLOC×6 = 13
LEARNING_RATE = 0.001
GAMMA = 0.99             # Discount factor
EPSILON_START = 1.0      # Initial exploration rate
EPSILON_DECAY = 0.99      # Decay rate PER UPDATE (not per episode)
EPSILON_MIN = 0.01        # Minimum exploration rate
BATCH_SIZE = 64          # Mini-batch size for training
MEMORY_SIZE = 10000      # Replay memory capacity

REWARD_SATISFIED = 10.0        # SFC completed successfully within E2E delay
REWARD_PARTIAL = +1.0          # tiến triển 1 VNF
REWARD_DROPPED = -10.0         # SFC dropped due to delay constraint violation
REWARD_INVALID = -5.0         # Invalid action (critical error: no resources)
REWARD_UNINSTALL_REQ = -3.0   # Attempting to uninstall a required VNF
REWARD_WAIT = -0.1             # Wait action (neutral)

# Priority P = P1 + P2 + P3, where:
# - P1 = TE^s - D^s (time pressure)
# - P2 = Location affinity (same DC vs different DC)
# - P3 = Urgency factor (when remaining time < threshold)

PRIORITY_P2_SAME_DC = 50.0    # Bonus if previous VNF is on same DC
PRIORITY_P2_DIFF_DC = -20.0   # Penalty if previous VNF is on different DC
URGENCY_THRESHOLD = 20        # Threshold (ms) for urgent requests
URGENCY_CONSTANT_C = 100.0    # Constant C in formula P3 = C / (D^s - TE^s + ε)
EPSILON = 1e-5                # Small value to prevent division by zero

WEIGHTS_FILE = 'sfc_dqn_weights.weights.h5'
TEST_EPSILON = 0.0            # No exploration during testing
TEST_EPISODES = 5            # Number of test episodes per configuration
TEST_FIG3_DCS = [2, 4, 6, 8]  # DC counts for reconfigurability test (Fig 3)