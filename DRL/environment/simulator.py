# environment/simulator.py
import config

class Simulator:
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time_ms = 0
        
    def reset(self):
        self.sim_time_ms = 0
        # Generate initial traffic
        self.manager.generate_requests(self.sim_time_ms, len(self.dcs))

    def advance_time(self):
        """
        Tăng thời gian thêm 1ms (Time Step).
        Return: Penalty nếu có request bị drop trong bước này.
        """
        self.sim_time_ms += config.TIME_STEP
        drop_penalty = 0.0

        # 1. Update VNFs (Processing progress)
        for dc in self.dcs:
            for v in dc.installed_vnfs:
                v.tick()

        # 2. Update Requests (Time elapsed check)
        # Lưu lại danh sách request ID đang active để kiểm tra drop
        pre_drop_check_ids = {r.id for r in self.manager.active_requests}
        
        for req in self.manager.active_requests:
            req.update_time() # Có thể set req.is_dropped = True
            
        # 3. Clean up (Move completed/dropped to history)
        self.manager.clean_requests()
        
        # 4. Calculate Penalty for newly dropped requests
        # (Những request vừa bị chuyển sang dropped_history)
        if self.manager.dropped_history:
            # Lấy những request vừa bị drop trong tick này
            # Logic đơn giản: đếm số lượng dropped tăng lên (nhưng chính xác hơn là check flag)
            # Ở đây ta giả định REWARD_DROPPED áp dụng cho mỗi request bị drop
            pass 
        
        # Để đơn giản, ta tính penalty dựa trên số lượng drop mới phát hiện
        # (Logic này đã được xử lý bên trong clean_requests gián tiếp, nhưng để tính reward cần cẩn thận)
        # Cách tốt nhất: Check flag is_dropped trong vòng lặp step 2
        for req in self.manager.active_requests: # List này chưa clean
             pass # Đã xử lý ở hàm clean.
             
        # Re-implement step 2 & 4 combined for accuracy:
        newly_dropped = 0
        for req in self.manager.active_requests:
            was_dropped = req.is_dropped
            req.update_time()
            if not was_dropped and req.is_dropped:
                newly_dropped += 1
                
        self.manager.clean_requests()
        drop_penalty = newly_dropped * config.REWARD_DROPPED

        # 5. Generate New Traffic
        if (self.sim_time_ms < config.TRAFFIC_STOP_TIME and 
            self.sim_time_ms % config.TRAFFIC_GEN_INTERVAL == 0):
            self.manager.generate_requests(self.sim_time_ms, len(self.dcs))
            
        return drop_penalty

    def is_done(self):
        # Stop if time exceeded max OR (traffic stopped AND no active requests)
        if self.sim_time_ms >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        if (self.sim_time_ms > config.TRAFFIC_STOP_TIME and 
            len(self.manager.active_requests) == 0):
            return True
        return False