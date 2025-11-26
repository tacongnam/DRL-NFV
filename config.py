import numpy as np

# Network Configuration
NUM_DCS = 6
LINK_BW = 1000
VNFS = ['NAT', 'FW', 'VOC', 'TM', 'WO', 'IDPS']
NUM_VNF_TYPES = len(VNFS)
    
# DC Resources
DC_CPU_RANGE = (12, 120)
DC_RAM = 256
DC_STORAGE = 2000

# SFC Profiles (Paper Table I)
SFC_PROFILES = {
    'CG': {'chain': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'], 'bw': 4, 'delay': 80},
    'AR': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 100, 'delay': 10},
    'VoIP': {'chain': ['NAT', 'FW', 'TM', 'FW', 'NAT'], 'bw': 0.064, 'delay': 100},
    'VS': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 4, 'delay': 100},
    'MIoT': {'chain': ['NAT', 'FW', 'IDPS'], 'bw': 1, 'delay': 5},
    'Ind4.0': {'chain': ['NAT', 'FW'], 'bw': 70, 'delay': 8}
}

# VNF Resource Consumption (Paper/Standard approx)
VNF_REQ = {
    'NAT': {'cpu': 2, 'ram': 2, 'sto': 10},
    'FW': {'cpu': 4, 'ram': 4, 'sto': 20},
    'VOC': {'cpu': 8, 'ram': 8, 'sto': 50},
    'TM': {'cpu': 2, 'ram': 2, 'sto': 10},
    'WO': {'cpu': 4, 'ram': 4, 'sto': 20},
    'IDPS': {'cpu': 8, 'ram': 8, 'sto': 50}
}

# Hyperparameters
STATE_DIM_DC = 3 + NUM_VNF_TYPES # CPU, RAM, BW, Installed_VNFs
LATENT_DIM = 16
ACTION_SPACE = 2 * NUM_VNF_TYPES + 1 # Install(N), Uninstall(N), Wait(1)

BATCH_SIZE = 64
GAMMA = 0.95
LR_VAE = 1e-3
LR_DQN = 5e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99