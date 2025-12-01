import numpy as np

VNF_TYPES = ['NAT', 'FW', 'VOC', 'TM', 'WO', 'IDPS']
SFC_TYPES = ['CG', 'AR', 'VS', 'VoIP', 'MIoT', 'Ind4.0']

SFC_CHARACTERISTICS = {
    'CG': {'chain': ['NAT', 'FW', 'VOC', 'TM'], 'bw': 30, 'delay': 50, 'bundle_size': (5, 15)},
    'AR': {'chain': ['NAT', 'FW', 'IDPS', 'TM'], 'bw': 25, 'delay': 20, 'bundle_size': (3, 10)},
    'VS': {'chain': ['NAT', 'FW', 'VOC'], 'bw': 20, 'delay': 100, 'bundle_size': (10, 20)},
    'VoIP': {'chain': ['NAT', 'FW'], 'bw': 5, 'delay': 150, 'bundle_size': (15, 30)},
    'MIoT': {'chain': ['NAT', 'FW', 'IDPS'], 'bw': 10, 'delay': 30, 'bundle_size': (8, 18)},
    'Ind4.0': {'chain': ['NAT', 'FW', 'WO', 'IDPS'], 'bw': 15, 'delay': 25, 'bundle_size': (5, 12)}
}

VNF_REQUIREMENTS = {
    'NAT': {'cpu': 2.0, 'ram': 4, 'storage': 10, 'proc_time': 2},
    'FW': {'cpu': 3.0, 'ram': 8, 'storage': 15, 'proc_time': 3},
    'VOC': {'cpu': 4.0, 'ram': 16, 'storage': 20, 'proc_time': 5},
    'TM': {'cpu': 2.5, 'ram': 6, 'storage': 12, 'proc_time': 2},
    'WO': {'cpu': 3.5, 'ram': 12, 'storage': 18, 'proc_time': 4},
    'IDPS': {'cpu': 4.5, 'ram': 20, 'storage': 25, 'proc_time': 6}
}

DC_CONFIG = {
    'storage_capacity': 2000,
    'cpu_range': (12, 120),
    'ram_capacity': 256,
    'link_bandwidth': 1000
}

TRAINING_CONFIG = {
    'num_updates': 350,
    'episodes_per_update': 20,
    'max_actions_per_step': 100,
    'step_duration': 1,
    'action_inference_time': 0.01,
    'request_generation_interval': 4,
    'gamma': 0.95,
    'learning_rate': 0.0001,
    'batch_size': 64,
    'memory_size': 100000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 10
}

REWARD_CONFIG = {
    'sfc_satisfied': 2.0,
    'sfc_dropped': -1.5,
    'invalid_action': -1.0,
    'uninstall_required': -0.5
}

PRIORITY_CONFIG = {
    'urgency_constant': 1000,
    'urgency_threshold': 10
}

SPEED_OF_LIGHT = 3e8