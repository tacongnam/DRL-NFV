import numpy as np
import config
from envs.base_env import SFCBaseEnv
from envs.vae_selector import VAEDCSelector
from envs.observer import Observer

class VAEEnv(SFCBaseEnv):
    """Environment with GenAI DC selection"""
    
    def __init__(self, vae_model=None, data_collection_mode=False, 
                 graph=None, dcs=None, requests_data=None):
        # Call parent with graph data
        super().__init__(graph=graph, dcs=dcs, requests_data=requests_data)
        
        # GenAI components
        self.vae_model = vae_model
        self.dc_selector = VAEDCSelector(vae_model) if vae_model else None
        self.data_collection_mode = data_collection_mode
        
        # Data collection specific
        self.all_dc_prev_states = None
        self.dc_prev_states = {}

    def _update_dc_order(self):
        """Update DC selection: Random (Collect) or VAE (Inference)"""
        if self.data_collection_mode or self.dc_selector is None:
            # Random selection
            self.dc_order = list(range(len(self.dcs)))
            np.random.shuffle(self.dc_order)
        else:
            # GenAI selection
            self.dc_order = self.dc_selector.get_dc_ranking(self.dcs, self.sfc_manager)

    def _reset_specific_state(self):
        """Clear prev states when reset"""
        self.dc_prev_states = {}
        self.all_dc_prev_states = None

    def _pre_action_hook(self):
        """Snapshot DC states before action (for data collection)"""
        if self.data_collection_mode:
            active_reqs = self.sfc_manager.active_requests
            global_stats = Observer.precompute_global_stats(
                self.sfc_manager, active_reqs
            )

            self.all_dcs_prev_states = {
                dc.id: Observer.get_dc_state(dc, self.sfc_manager, global_stats)
                for dc in self.dcs
            }

            curr_dc_id = self.dc_order[self.current_dc_idx]
            if curr_dc_id in self.all_dcs_prev_states:
                 self.dc_prev_states[curr_dc_id] = self.all_dcs_prev_states[curr_dc_id]

    def _post_step_info_hook(self, info):
        """Add data collection info"""
        if self.data_collection_mode:
            info['dc_prev_states'] = self.dc_prev_states

    def get_dc_transitions(self):
        """Get DC transitions for VAE training"""
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
        
        self.dc_prev_states = {}
        
        return transitions