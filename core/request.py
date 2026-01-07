class Request:
    def __init__(self, id, arrival_time, source, destination, bandwidth, vnf_chain, max_delay):
        self.id = id
        self.arrival_time = arrival_time
        self.source = source
        self.destination = destination
        self.bandwidth = bandwidth
        self.chain = vnf_chain
        self.max_delay = max_delay

        self.current_vnf_idx = 0
        self.elapsed_time = 0.0
        self.placed_vnfs = []
        self.allocated_paths = []
        
        self.is_completed = False
        self.is_dropped = False

    def get_next_vnf(self):
        if not self.is_completed and self.current_vnf_idx < len(self.chain):
            return self.chain[self.current_vnf_idx]
        return None
    
    def get_last_placed_dc(self):
        if not self.placed_vnfs:
            return self.source
        return self.placed_vnfs[-1][0]
    
    def get_remaining_time(self):
        return max(0.0, self.max_delay - self.elapsed_time)

    def advance_chain(self, dc_id, prop_delay, proc_delay, vnf_instance):
        if self.is_completed:
            return

        vnf_type = self.chain[self.current_vnf_idx]
        self.placed_vnfs.append((dc_id, vnf_type))
        
        total_delay = prop_delay + proc_delay
        self.elapsed_time += total_delay
        
        self.current_vnf_idx += 1
        
    def check_completion(self):
        if self.current_vnf_idx >= len(self.chain):
            self.is_completed = True
            return True
        return False

    def update_time(self, dt):
        if not self.is_completed and not self.is_dropped:
            self.elapsed_time += dt
            if self.elapsed_time > self.max_delay:
                self.is_dropped = True