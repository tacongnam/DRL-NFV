import config

class SFCRequest:
    def __init__(self, req_id, vnf_chain, source, destination, arrival_time, bandwidth, max_delay, sfc_type="Unknown"):
        self.id = req_id
        self.chain = vnf_chain[:]
        self.type = sfc_type  # Optional label for analysis
        self.current_vnf_index = 0
        
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        self.bandwidth = bandwidth
        self.max_delay = max_delay
        
        self.elapsed_time = 0.0
        self.is_dropped = False
        self.is_completed = False
        
        self.placed_vnfs = []
        self.total_propagation_delay = 0.0
        self.total_processing_delay = 0.0
        self.remaining_processing_time = 0.0
        
        self.allocated_paths = []

    @property
    def is_placed(self):
        return self.current_vnf_index >= len(self.chain)

    def get_next_vnf(self):
        return self.chain[self.current_vnf_index] if not self.is_placed else None

    def get_last_placed_dc(self):
        return self.placed_vnfs[-1][1] if self.placed_vnfs else self.source

    def advance_chain(self, dc_id, prop_delay, proc_delay, vnf_instance):
        if self.is_completed or self.is_placed:
            return
        
        vnf_type = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf_type, dc_id, prop_delay, proc_delay))
        
        self.total_propagation_delay += prop_delay
        self.remaining_processing_time += proc_delay
        self.total_processing_delay += proc_delay
        
        self.current_vnf_index += 1

    def check_completion(self):
        if self.is_placed and self.remaining_processing_time <= 0:
            self.is_completed = True

    def update_time(self, time_step=None):
        if self.is_completed or self.is_dropped:
            return
        
        step = time_step if time_step is not None else config.TIME_STEP
        self.elapsed_time += step
        
        if self.remaining_processing_time > 0:
            self.remaining_processing_time = max(0, self.remaining_processing_time - step)
        
        self.check_completion()
        
        if not self.is_completed and self.elapsed_time > self.max_delay:
            self.is_dropped = True

    def get_total_e2e_delay(self):
        return self.total_propagation_delay + self.total_processing_delay

    def get_remaining_time(self):
        return max(0, self.max_delay - self.elapsed_time)