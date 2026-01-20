from envs.base_env import SFCBaseEnv
from envs.priority import PriorityManager

class DRLEnv(SFCBaseEnv):
    def __init__(self, graph=None, dcs=None, requests_data=None):
        super().__init__(graph=graph, dcs=dcs, requests_data=requests_data)
        self.priority_manager = None

    def _after_init_components(self):
        self.priority_manager = PriorityManager(self.topology)

    def _update_dc_order(self):
        if self.priority_manager:
            self.dc_order = self.priority_manager.get_dc_priority_order(
                self.dcs, 
                self.sfc_manager.active_requests
            )
        else:
            self.dc_order = [dc.id for dc in self.dcs if dc.is_server]

Env = DRLEnv