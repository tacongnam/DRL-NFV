import random
import config

class TrafficGenerator:
    def __init__(self):
        self.sfc_keys = list(config.SFC_TYPES.keys())
    
    def generate_request(self):
        """
        [Paper Section V.A]: SFC requests are generated randomly.
        """
        sfc_type = random.choice(self.sfc_keys)
        req_data = config.SFC_TYPES[sfc_type].copy()
        # Thêm ID để tracking
        req_data['id'] = random.randint(10000, 99999)
        return sfc_type, req_data