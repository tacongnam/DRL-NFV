import config

class Simulator:
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time = 0
        
    def reset(self):
        self.sim_time = 0

    def advance_time(self):
        """
        Tiến thời gian mô phỏng thêm TIME_STEP.
        Cập nhật VNF và Request.
        Trả về penalty dựa trên số lượng request bị drop trong bước này.
        """
        self.sim_time += config.TIME_STEP
        
        # 1. Update VNFs (giảm thời gian xử lý)
        for dc in self.dcs:
            if dc.is_server:
                for vnf in dc.installed_vnfs:
                    vnf.tick()
        
        # 2. Update Requests (kiểm tra timeout)
        pre_dropped_count = len(self.manager.dropped_history)
        
        # Kích hoạt request mới nếu đến giờ
        self.manager.activate_new_requests(self.sim_time)
        
        # Cập nhật trạng thái request
        self.manager.step(config.TIME_STEP)

        # 3. Calculate Penalty
        newly_dropped = len(self.manager.dropped_history) - pre_dropped_count
        penalty = newly_dropped * config.REWARD_DROPPED
        
        return penalty

    def is_done(self):
        # Điều kiện dừng: Hết thời gian HOẶC Hết request (cả active và future)
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        if len(self.manager.active_requests) == 0:
            # Kiểm tra xem còn request nào chưa đến giờ không
            if self.manager.request_cursor >= len(self.manager.all_requests):
                return True
        
        return False