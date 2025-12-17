import numpy as np
import config
from envs.base_env import SFCBaseEnv
from envs.vae_selector import VAEDCSelector
from envs.observer import Observer

class VAEEnv(SFCBaseEnv):
    """Environment với GenAI DC selection"""
    
    def __init__(self, genai_model=None, data_collection_mode=False):
        super().__init__()
        
        # GenAI components
        self.genai_model = genai_model
        self.dc_selector = VAEDCSelector(genai_model) if genai_model else None
        self.data_collection_mode = data_collection_mode
        
        # Data collection specific
        self.all_dc_prev_states = None
        self.dc_prev_states = {} # Giữ lại để tương thích code cũ nếu cần

    def _update_dc_order(self):
        """Cập nhật DC selection order: Random (Collect) hoặc VAE (Inference)"""
        if self.data_collection_mode or self.dc_selector is None:
            # Random selection
            self.dc_order = list(range(len(self.dcs)))
            np.random.shuffle(self.dc_order)
        else:
            # GenAI selection
            self.dc_order = self.dc_selector.get_dc_ranking(self.dcs, self.sfc_manager)

    def _reset_specific_state(self):
        """Clear prev states khi reset"""
        self.dc_prev_states = {}
        self.all_dc_prev_states = None

    def _pre_action_hook(self):
        """Snapshot trạng thái TOÀN BỘ DC trước khi action (cho data collection)"""
        if self.data_collection_mode:
            # Tối ưu: Lấy request active 1 lần
            active_reqs = self.sfc_manager.active_requests
            global_stats = Observer.precompute_global_stats(
                self.sfc_manager, active_reqs
            )

            self.all_dcs_prev_states = {
                dc.id: Observer.get_dc_state(dc, self.sfc_manager, global_stats)
                for dc in self.dcs
            }

            # Cập nhật dc_prev_states (nếu logic cũ còn dùng)
            curr_dc_id = self.dc_order[self.current_dc_idx]
            if curr_dc_id in self.all_dcs_prev_states:
                 self.dc_prev_states[curr_dc_id] = self.all_dcs_prev_states[curr_dc_id]

    def _post_step_info_hook(self, info):
        """Thêm thông tin data collection vào info"""
        if self.data_collection_mode:
            info['dc_prev_states'] = self.dc_prev_states

    def get_dc_transitions(self):
        """Hàm riêng của VAEEnv để lấy dữ liệu training"""
        transitions = []
        if not hasattr(self, 'all_dcs_prev_states') or not self.all_dcs_prev_states:
            return []
        
        active_reqs = Observer.precompute_global_stats(self.sfc_manager)
        
        for dc in self.dcs:
            if dc.id in self.all_dcs_prev_states:
                prev_state = self.all_dcs_prev_states[dc.id]
                curr_state = Observer.get_dc_state(dc, self.sfc_manager, active_reqs)
                value = Observer.calculate_dc_value(dc, self.sfc_manager, prev_state, active_reqs)
                
                transitions.append((dc.id, prev_state, curr_state, value))
        
        # Clear (tùy logic, ở đây giữ nguyên theo code cũ)
        self.dc_prev_states = {} 
        
        return transitions