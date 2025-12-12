# environment/simulator.py
import config

class Simulator:
    """Quản lý thời gian simulation và logic sinh request"""
    
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time = 0  # ms
        self.has_generated_initial = False
        
    def reset(self):
        """Reset simulator về trạng thái ban đầu"""
        self.sim_time = 0
        self.has_generated_initial = False

    def advance_time(self):
        """
        Tăng thời gian 1 time step và xử lý các sự kiện
        
        Returns:
            penalty: Penalty từ các request bị drop
        """
        self.sim_time += config.TIME_STEP
        penalty = 0.0
        
        # 1. Update VNFs processing (tick trước)
        for dc in self.dcs:
            for vnf in dc.installed_vnfs:
                vnf.tick()
        
        # 2. Update requests và phát hiện drop
        newly_dropped = []
        for req in self.manager.active_requests[:]:  # Copy list to avoid modification during iteration
            was_dropped = req.is_dropped
            was_completed = req.is_completed
            
            # Update time (includes check_completion)
            req.update_time()
            
            # Detect newly dropped
            if not was_dropped and req.is_dropped:
                newly_dropped.append(req)
        
        # 3. Calculate penalty TRƯỚC khi clean
        penalty = len(newly_dropped) * config.REWARD_DROPPED
        
        # 4. Clean completed/dropped requests
        self.manager.clean_requests()
        
        # 5. Generate new traffic
        if self.sim_time < config.TRAFFIC_STOP_TIME:
            if not self.has_generated_initial:
                # Sinh request đầu tiên
                self.manager.generate_requests(self.sim_time, len(self.dcs))
                self.has_generated_initial = True
            elif self.sim_time % config.TRAFFIC_GEN_INTERVAL == 0:
                # Sinh request định kỳ
                self.manager.generate_requests(self.sim_time, len(self.dcs))
        
        return penalty

    def is_done(self):
        """
        Kiểm tra episode có kết thúc chưa
        
        Điều kiện kết thúc:
        1. Vượt quá thời gian tối đa, HOẶC
        2. Đã dừng sinh request VÀ không còn request active
        """
        # Điều kiện 1: Vượt thời gian max
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        # Điều kiện 2: Dừng sinh request và hết request
        if self.sim_time > config.TRAFFIC_STOP_TIME:
            if len(self.manager.active_requests) == 0:
                return True
        
        return False
    
    def should_generate_initial_traffic(self):
        """Kiểm tra có nên sinh traffic ban đầu không"""
        return not self.has_generated_initial