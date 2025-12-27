# environment/simulator.py
from DRL import config

class Simulator:
    """Manages simulation time and request activation logic"""
    
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time = 0  # ms
        
    def reset(self):
        """Reset simulator to initial state"""
        self.sim_time = 0

    def advance_time(self):
        """
        Advance time by one time step and process events.
        
        Returns:
            penalty: Penalty from dropped requests
        """
        self.sim_time += config.TIME_STEP
        penalty = 0.0
        
        # 1. Update VNFs processing (tick first)
        for dc in self.dcs:
            for vnf in dc.installed_vnfs:
                vnf.tick()
        
        # 2. Update requests and detect drops
        newly_dropped = []
        for req in self.manager.active_requests[:]:  # Copy list to avoid modification during iteration
            was_dropped = req.is_dropped
            was_completed = req.is_completed
            
            # Update time (includes check_completion)
            req.update_time()
            
            # Detect newly dropped
            if not was_dropped and req.is_dropped:
                newly_dropped.append(req)
        
        # 3. Calculate penalty BEFORE cleaning
        penalty = len(newly_dropped) * config.REWARD_DROPPED
        
        # 4. Clean completed/dropped requests
        self.manager.clean_requests()
        
        # 5. Activate new requests that arrive at current time
        self.manager.activate_requests_at_time(self.sim_time)
        
        return penalty

    def is_done(self):
        """
        Check if episode is finished.
        
        Termination conditions:
        1. Exceeded maximum time, OR
        2. All requests have been processed (no more active or pending)
        """
        # Condition 1: Exceeded max time
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        # Condition 2: No more requests to process
        # Check if there are any active requests or future arrivals
        if len(self.manager.active_requests) == 0:
            # Check if there are any requests that haven't arrived yet
            future_requests = [r for r in self.manager.all_requests 
                             if r.arrival_time > self.sim_time]
            if len(future_requests) == 0:
                return True
        
        return False