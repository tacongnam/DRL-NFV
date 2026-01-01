import numpy as np
import config
from envs.base_env import SFCBaseEnv
from envs.vae_selector import VAEDCSelector
from envs.observer import Observer

class VAEEnv(SFCBaseEnv):
    def __init__(self, vae_model=None, data_collection_mode=False, 
                 graph=None, dcs=None, requests_data=None):
        super().__init__(graph=graph, dcs=dcs, requests_data=requests_data)
        
        self.vae_model = vae_model
        self.dc_selector = VAEDCSelector(vae_model) if vae_model else None
        self.data_collection_mode = data_collection_mode
        
        self.all_dc_prev_states = None
        self.dc_prev_states = {}

    def _update_dc_order(self):
        server_ids = [dc.id for dc in self.dcs if dc.is_server]
        
        if self.data_collection_mode or self.dc_selector is None:
            np.random.shuffle(server_ids)
            self.dc_order = server_ids
        else:
            server_dcs = [dc for dc in self.dcs if dc.is_server]
            ranked_indices = self.dc_selector.get_dc_ranking(server_dcs, self.sfc_manager)
            self.dc_order = [server_dcs[i].id for i in ranked_indices]

    def _reset_specific_state(self):
        self.dc_prev_states = {}
        self.all_dc_prev_states = None

    def _pre_action_hook(self):
        if self.data_collection_mode:
            active_reqs = self.sfc_manager.active_requests
            global_stats = Observer.precompute_global_stats(
                self.sfc_manager, active_reqs
            )

            self.all_dc_prev_states = {
                dc.id: Observer.get_dc_state(dc, self.sfc_manager, global_stats)
                for dc in self.dcs if dc.is_server
            }

            curr_dc_id = self.dc_order[self.current_dc_idx]
            if curr_dc_id in self.all_dc_prev_states:
                self.dc_prev_states[curr_dc_id] = self.all_dc_prev_states[curr_dc_id]

    def _post_step_info_hook(self, info):
        if self.data_collection_mode:
            info['dc_prev_states'] = self.dc_prev_states

    def get_dc_transitions(self):
        transitions = []
        if not hasattr(self, 'all_dc_prev_states') or not self.all_dc_prev_states:
            return []
        
        active_reqs = Observer.precompute_global_stats(self.sfc_manager)
        
        for dc in self.dcs:
            if dc.is_server and dc.id in self.all_dc_prev_states:
                prev_state = self.all_dc_prev_states[dc.id]
                curr_state = Observer.get_dc_state(dc, self.sfc_manager, active_reqs)
                value = Observer.calculate_dc_value(dc, self.sfc_manager, prev_state, active_reqs)
                
                transitions.append((dc.id, prev_state, curr_state, value))
        
        self.dc_prev_states = {}
        
        return transitions