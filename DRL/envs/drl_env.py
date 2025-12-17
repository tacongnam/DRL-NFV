from envs.base_env import SFCBaseEnv
from envs.priority import PriorityManager
from envs.observer import Observer

class Env(SFCBaseEnv):
    """Standard DRL Environment với Priority-based selection"""
    
    def __init__(self):
        super().__init__()
        self.priority_manager = None # Sẽ init trong hook

    def _after_init_components(self):
        """Init Priority Manager sau khi topology đã có"""
        self.priority_manager = PriorityManager(self.topology)

    def _update_dc_order(self):
        """Cập nhật DC order dựa trên Priority Rule"""
        if self.priority_manager:
            self.dc_order = self.priority_manager.get_dc_priority_order(
                self.dcs, 
                self.sfc_manager.active_requests
            )
        else:
            # Fallback nếu chưa init
            self.dc_order = [dc.id for dc in self.dcs]

    def _get_obs(self):
        """
        Override hàm của cha để lấy Tuple State cho DRL
        thay vì Flat State cho VAE.
        """
        if not self.dc_order:
            self._update_dc_order()
            
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        
        # GỌI HÀM MỚI
        return Observer.get_drl_observation(curr_dc, self.sfc_manager)