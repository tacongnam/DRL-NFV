SFC_TYPES = {
    'CG': {  
        'vnfs': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'],
        'bw': 4,      
        'delay': 80, 
        'bundle': (40, 55)
    },

    'AR': { 
        'vnfs': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'],
        'bw': 100,
        'delay': 10,
        'bundle': (1, 4)
    },

    'VoIP': {
        'vnfs': ['NAT', 'FW', 'TM'], 
        'bw': 0.064,
        'delay': 100,
        'bundle': (100, 200)
    },

    'VS': { 
        'vnfs': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'],
        'bw': 4,
        'delay': 100,
        'bundle': (50, 100)
    },

    'MIoT': {
        'vnfs': ['NAT', 'FW', 'IDPS'],
        'bw': (1, 50),
        'delay': 5,
        'bundle': (10, 15)
    },

    'Ind4.0': {
        'vnfs': ['NAT', 'FW'],
        'bw': 70,
        'delay': 8,
        'bundle': (1, 4)
    }
}

VNF_SPECS = {
    'NAT': {
        'cpu': 1,
        'ram': 4,
        'storage': 7,
        'proc_time': 0.06 
    },
    'FW': {
        'cpu': 9,
        'ram': 5,
        'storage': 1,
        'proc_time': 0.03
    },
    'VOC': {
        'cpu': 5,
        'ram': 11,
        'storage': 13,
        'proc_time': 0.11
    },
    'TM': {
        'cpu': 13,
        'ram': 7,
        'storage': 7,
        'proc_time': 0.07
    },
    'WO': {
        'cpu': 5,
        'ram': 2,
        'storage': 5,
        'proc_time': 0.08
    },
    'IDPS': {
        'cpu': 11,
        'ram': 15,
        'storage': 2,
        'proc_time': 0.02
    }
}

VNF_LIST = ['NAT', 'FW', 'VOC', 'TM', 'WO', 'IDPS']

DC_CONFIG = {
    'cpu_range': (12, 120),
    'ram': 256,
    'storage': 2048,
    'link_bw': 1000
}

DRL_CONFIG = {
    'updates': 350,
    'episodes_per_update': 20,
    'actions_per_step': 100,
    'step_duration': 1,
    'action_inference_time': 0.01,
    'request_interval': 2,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'memory_size': 10000,
    'target_update_freq': 10
}

REWARD_CONFIG = {
    'sfc_satisfied': 5.0,
    'sfc_dropped': -2.0,
    'invalid_action': -0.1,
    'uninstall_required': -0.05,
    'default': 0.0
}

PRIORITY_CONFIG = {
    'urgency_constant': 100,
    'urgency_threshold': 10,
    'epsilon': 1e-6
}

LIGHT_SPEED = 3e8