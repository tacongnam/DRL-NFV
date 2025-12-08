import config

class SFCRequest:
    def __init__(self, req_id, sfc_type, source, destination, arrival_time):
        self.id = req_id
        self.type = sfc_type
        self.specs = config.SFC_SPECS[sfc_type]
        self.chain = self.specs['chain'][:]
        self.current_vnf_index = 0
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        
        self.max_delay = self.specs['delay'] # D^s
        self.elapsed_time = 0                # TE^s
        self.is_dropped = False
        self.is_completed = False
        
        self.placed_vnfs = []                # List of (vnf_type, dc_id) tuples

    def get_next_vnf(self):
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None

    def advance_chain(self, dc_id):
        vnf = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf, dc_id))
        self.current_vnf_index += 1
        if self.current_vnf_index >= len(self.chain):
            self.is_completed = True

    def update_time(self):
        self.elapsed_time += config.TIME_STEP
        # Bài báo: Dropped if propagation + processing > D^s. 
        # Ở đây ta check tổng thời gian đã trôi qua.
        if self.elapsed_time > self.max_delay:
            self.is_dropped = True
