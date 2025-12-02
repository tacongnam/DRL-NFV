import numpy as np

SFC_TYPES = {
    'CG': {'vnfs': ['NAT', 'FW', 'VOC', 'TM'], 'bw': 50, 'delay': 50, 'bundle': (10, 20)},
    'AR': {'vnfs': ['NAT', 'FW', 'VOC'], 'bw': 30, 'delay': 20, 'bundle': (15, 25)},
    'VS': {'vnfs': ['NAT', 'FW', 'VOC', 'TM'], 'bw': 40, 'delay': 100, 'bundle': (20, 30)},
    'VoIP': {'vnfs': ['NAT', 'FW'], 'bw': 10, 'delay': 150, 'bundle': (25, 35)},
    'MIoT': {'vnfs': ['NAT', 'FW', 'IDPS'], 'bw': 5, 'delay': 30, 'bundle': (30, 40)},
    'Ind4.0': {'vnfs': ['NAT', 'FW', 'IDPS', 'WO'], 'bw': 20, 'delay': 25, 'bundle': (20, 30)}
}

VNF_SPECS = {
    'NAT': {'cpu': 2, 'ram': 4, 'storage': 10, 'proc_time': 5},
    'FW': {'cpu': 3, 'ram': 6, 'storage': 15, 'proc_time': 8},
    'VOC': {'cpu': 4, 'ram': 8, 'storage': 20, 'proc_time': 12},
    'TM': {'cpu': 2, 'ram': 4, 'storage': 10, 'proc_time': 6},
    'WO': {'cpu': 3, 'ram': 6, 'storage': 15, 'proc_time': 10},
    'IDPS': {'cpu': 5, 'ram': 10, 'storage': 25, 'proc_time': 15}
}

VNF_LIST = ['NAT', 'FW', 'VOC', 'TM', 'WO', 'IDPS']

DC_CONFIG = {
    'cpu_range': (12, 120),
    'ram': 256,
    'storage': 2000,
    'link_bw': 1000
}

DRL_CONFIG = {
    'updates': 350,
    'episodes_per_update': 20,
    'actions_per_step': 100,
    'step_duration': 1,
    'action_inference_time': 0.01,
    'request_interval': 4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.0001,
    'batch_size': 64,
    'memory_size': 10000,
    'target_update_freq': 10
}

REWARD_CONFIG = {
    'sfc_satisfied': 2.0,
    'sfc_dropped': -1.5,
    'invalid_action': -1.0,
    'uninstall_required': -0.5,
    'default': 0.0
}

PRIORITY_CONFIG = {
    'urgency_constant': 100,
    'urgency_threshold': 10,
    'epsilon': 1e-6
}

LIGHT_SPEED = 3e8