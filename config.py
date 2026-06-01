NODE_DC = 0
NODE_SWITCH = 1
RESOURCE_TYPE = ["mem", "cpu", "ram"]
TIMESTEP = 0.1

LATENT_DIM = 8
MAX_DCS = 60
VGAE_DIR = "models/vgae_pretrained"
PLACER_DIR = "models/placer"
VGAE_WEIGHTS_FILE = "vgae_weights.npy"
PLACER_WEIGHTS_FILE = "placer_dqn_weights.npy"
PLACER_WEIGHT_NET_FILE = "placer_weight_net_weights.npy"

DEFAULT_VGAE_EPOCHS = 60
DEFAULT_PLACER_EPISODES = 60
DEFAULT_REQUEST_PCT = 0

DRL_PENALTY_DROP = 0.5
DRL_R_BASE_LL = 1.0
DRL_BATCH_SIZE = 32
DRL_TARGET_SYNC = 40
DRL_VGAE_TRAIN_FREQ = 500
DRL_VGAE_EPOCHS = 3
DRL_MAX_GRAPH_CACHE = 500

EPSILON_MAX = 0.9
EPSILON_MIN = 0.1
EPSILON_WARMUP = 0.1

DRL_LL_ALPHA = 0.5
DRL_LL_BETA = 1.0

ROUTING_K_PATHS = 3
 
ROUTING_DELAY_WEIGHT    = 0.4  # w_delay : tổng delay dọc path
ROUTING_BW_WEIGHT       = 0.3  # w_bw    : mean exponential BW pressure
ROUTING_PRESSURE_WEIGHT = 0.2  # w_mm1   : mean M/M/1 queuing pressure
ROUTING_HOP_WEIGHT      = 0.1  # w_hops  : số hop (tránh path quá dài)

HRL_VGAE_FINETUNE_LR   = 1e-5
HRL_VGAE_FINETUNE_FREQ = 500
HRL_VGAE_FINETUNE_EPOCHS = 1
HRL_VGAE_ONLINE        = False