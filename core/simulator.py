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
        
        for dc in self.dcs:
            if dc.is_server:
                for vnf in dc.installed_vnfs:
                    vnf.tick()
        
        pre_dropped_count = len(self.manager.dropped_history)
        
        self.manager.activate_new_requests(self.sim_time)
        self.manager.step(config.TIME_STEP)

        newly_dropped = len(self.manager.dropped_history) - pre_dropped_count
        
        if newly_dropped > 0:
            penalty = newly_dropped * config.REWARD_DROPPED
        else:
            penalty = 0.0
        
        return penalty

    def is_done(self):
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        if len(self.manager.active_requests) == 0:
            if self.manager.request_cursor >= len(self.manager.all_requests):
                return True
        
        return False