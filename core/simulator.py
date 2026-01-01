# environment/simulator.py
import config

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
            if dc.is_server:
                for vnf in dc.installed_vnfs:
                    vnf.tick()
        
        # 2. Update requests and detect drops
        pre_dropped = len(self.manager.dropped_history)
        self.manager.step(config.TIME_STEP)

        newly_dropped = len(self.manager.dropped_history) - pre_dropped
        penalty = newly_dropped * config.REWARD_DROPPED
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