import config

class Simulator:
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time = 0
        
    def reset(self):
        self.sim_time = 0

    def advance_time(self):
        self.sim_time += config.TIME_STEP
        penalty = 0.0
        
        # Update VNFs
        for dc in self.dcs:
            if dc.is_server:
                for vnf in dc.installed_vnfs:
                    vnf.tick()
        
        # Update requests
        pre_dropped = len(self.manager.dropped_history)
        self.manager.step(config.TIME_STEP)

        newly_dropped = len(self.manager.dropped_history) - pre_dropped
        penalty = newly_dropped * config.REWARD_DROPPED
        return penalty

    def is_done(self):
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        if len(self.manager.active_requests) == 0:
            future_requests = [r for r in self.manager.all_requests 
                             if r.arrival_time > self.sim_time]
            if len(future_requests) == 0:
                return True
        
        return False