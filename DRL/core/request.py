# spaces/request.py
from DRL import config

class SFCRequest:
    """Represents an SFC request loaded from data"""
    
    def __init__(self, req_id, vnf_chain, source, destination, arrival_time, bandwidth, max_delay):
        """
        Initialize SFC request with data from JSON.
        
        Args:
            req_id: Request ID
            vnf_chain: List of VNF type indices (e.g., [0, 1, 2])
            source: Source node ID
            destination: Destination node ID
            arrival_time: Request arrival time (ms)
            bandwidth: Required bandwidth
            max_delay: Maximum delay allowed (ms)
        """
        self.id = req_id
        self.chain = vnf_chain[:]  # VNF chain (list of VNF type indices)
        self.current_vnf_index = 0
        
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        self.bandwidth = bandwidth
        
        self.max_delay = max_delay  # ms
        self.elapsed_time = 0.0  # ms
        
        self.is_dropped = False
        self.is_completed = False
        self.all_vnfs_processed = False  # Flag to mark when all VNFs are processed
        
        # Store placed VNF info: [(vnf_type_idx, dc_id, propagation_delay, processing_delay)]
        self.placed_vnfs = []
        
        # Accumulated delays
        self.total_propagation_delay = 0.0  # ms
        self.total_processing_delay = 0.0   # ms
        
        # Track VNF instances processing this request
        self.processing_vnf_instances = []  # List of VNFInstance objects

    def get_next_vnf(self):
        """Get next VNF type index in chain that needs to be placed"""
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None

    def advance_chain(self, dc_id, prop_delay=0.0, proc_delay=0.0, vnf_instance=None):
        """
        Advance in chain after successfully placing VNF.
        Processing delay is calculated as startup_time + dc_delay (see vnf.py)
        
        Args:
            dc_id: ID of DC where VNF is placed
            prop_delay: Propagation delay from previous DC to this DC (ms)
            proc_delay: Processing delay of this VNF (startup_time + dc_delay) (ms)
            vnf_instance: VNFInstance object doing the processing
        """
        if self.is_completed:
            return
        
        vnf_type_idx = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf_type_idx, dc_id, prop_delay, proc_delay))
        
        # Accumulate delays
        self.total_propagation_delay += prop_delay
        self.total_processing_delay += proc_delay
        
        # Track VNF instance
        if vnf_instance:
            self.processing_vnf_instances.append(vnf_instance)
        
        self.current_vnf_index += 1
        
        # Check if all VNFs are placed (but not yet processed)
        if self.current_vnf_index >= len(self.chain):
            # All VNFs placed, waiting for processing
            pass

    def check_completion(self):
        """
        Check if all VNFs have finished processing.
        Only complete when:
        1. All VNFs in chain are placed
        2. All VNFs have finished processing (idle)
        """
        if self.is_completed or self.is_dropped:
            return
        
        # All VNFs placed?
        if self.current_vnf_index >= len(self.chain):
            # Check if all VNFs have finished processing
            all_idle = all(vnf.is_idle() for vnf in self.processing_vnf_instances)
            
            if all_idle:
                self.is_completed = True
                self.all_vnfs_processed = True

    def get_total_e2e_delay(self):
        """Calculate total E2E delay (propagation + processing)"""
        return self.total_propagation_delay + self.total_processing_delay

    def get_remaining_time(self):
        """Time remaining before being dropped"""
        return max(0, self.max_delay - self.elapsed_time)

    def update_time(self):
        """Update elapsed time"""
        if self.is_completed or self.is_dropped:
            return
        
        self.elapsed_time += config.TIME_STEP
        
        # Check completion before checking drop
        self.check_completion()
        
        # Check drop condition (only drop if not completed)
        if not self.is_completed and self.elapsed_time > self.max_delay:
            self.is_dropped = True

    def get_last_placed_dc(self):
        """Get DC ID of last placed VNF"""
        if self.placed_vnfs:
            return self.placed_vnfs[-1][1]
        return None