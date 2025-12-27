# spaces/vnf.py
from DRL import config

class VNFInstance:
    """Represents a VNF instance installed at a DC"""
    
    def __init__(self, vnf_type, dc_id):
        """
        Initialize VNF instance.
        
        Args:
            vnf_type: VNF type index (int)
            dc_id: Data center ID where VNF is deployed
        """
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.remaining_time = 0.0  # ms - remaining processing time
        self.assigned_sfc_id = None
        self.waiting_time = 0.0  # ms - waiting time before processing

    def is_idle(self):
        """Check if VNF is idle and available for assignment"""
        return self.remaining_time <= 0 and self.assigned_sfc_id is None

    def assign(self, sfc_id, dc_delay, waiting_time_ms=0.0):
        """
        Assign VNF to an SFC request.
        Processing time = startup_time[dc_id] + dc_delay
        
        Args:
            sfc_id: ID of the SFC request
            dc_delay: DC delay (ms)
            waiting_time_ms: Waiting time before processing starts (ms)
        """
        self.assigned_sfc_id = sfc_id
        
        # Get startup time from VNF_SPECS for this VNF type at this DC
        startup_time = config.VNF_SPECS[self.vnf_type]['startup_time'].get(self.dc_id, 0.0)
        
        # Calculate processing time: startup_time + DC delay
        processing_time = startup_time + dc_delay
        
        # Convert to ticks (ensure at least one time step)
        self.remaining_time = max(config.TIME_STEP, processing_time)
        self.waiting_time = waiting_time_ms

    def tick(self):
        """Update VNF state after each time step"""
        if self.remaining_time > 0:
            self.remaining_time = max(0, self.remaining_time - config.TIME_STEP)
            
            # When processing completes, release VNF
            if self.remaining_time <= 0:
                self.assigned_sfc_id = None
                self.waiting_time = 0.0
                self.remaining_time = 0.0

    def get_total_delay(self):
        """Return total delay (waiting + remaining processing time)"""
        return self.waiting_time + self.remaining_time
    
    def get_processing_time(self, dc_delay):
        """
        Calculate processing time for this VNF at its DC.
        Processing time = startup_time + dc_delay
        
        Args:
            dc_delay: DC delay (ms)
            
        Returns:
            Processing time in ms
        """
        startup_time = config.VNF_SPECS[self.vnf_type]['startup_time'].get(self.dc_id, 0.0)
        return startup_time + dc_delay