import gymnasium as gym
import numpy as np
from gymnasium import spaces

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils import *

class SFCEnvironment(gym.Env):
    def __init__(self, num_dcs=4):
        super().__init__()
        
        self.num_dcs = num_dcs
        self.num_vnf_types = len(VNF_TYPES)
        self.num_sfc_types = len(SFC_TYPES)
        
        self.action_space = spaces.Discrete(2 * self.num_vnf_types + 1)
        
        state1_dim = 2 * self.num_vnf_types + 2
        state2_dim = self.num_sfc_types * (1 + 2 * self.num_vnf_types)
        state3_dim = self.num_sfc_types * (4 + self.num_vnf_types)
        
        self.observation_space = spaces.Dict({
            'state1': spaces.Box(low=0, high=1, shape=(state1_dim,), dtype=np.float32),
            'state2': spaces.Box(low=0, high=1, shape=(state2_dim,), dtype=np.float32),
            'state3': spaces.Box(low=0, high=1, shape=(state3_dim,), dtype=np.float32)
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.dcs = create_dc_resources(self.num_dcs)
        self.distance_matrix = create_distance_matrix(self.num_dcs)
        self.link_bandwidth = np.full((self.num_dcs, self.num_dcs), DC_CONFIG['link_bandwidth'])
        self.link_bandwidth_used = np.zeros((self.num_dcs, self.num_dcs))
        
        self.sfc_requests = []
        self.current_time = 0
        self.current_dc_idx = 0
        self.total_accepted = 0
        self.total_dropped = 0
        self.total_generated = 0
        
        return self._get_state(), {}
    
    def _get_state(self):
        dc = self.dcs[self.current_dc_idx]
        
        state1 = []
        for vnf in VNF_TYPES:
            state1.append(dc['installed_vnfs'][vnf] / 10)
            state1.append(max(0, dc['installed_vnfs'][vnf] - dc['allocated_vnfs'][vnf]) / 10)
        state1.append(1 - dc['storage_used'] / dc['storage'])
        state1.append(1 - dc['cpu_used'] / dc['cpu'])
        
        state2 = []
        for sfc_type in SFC_TYPES:
            sfc_info = [0] * (1 + 2 * self.num_vnf_types)
            sfc_info[0] = SFC_TYPES.index(sfc_type) / len(SFC_TYPES)
            
            for req in self.sfc_requests:
                if req['type'] == sfc_type and req['status'] == 'pending':
                    for vnf_idx, vnf in enumerate(req['chain']):
                        if vnf_idx < len(req['allocated_vnfs']):
                            sfc_info[1 + VNF_TYPES.index(vnf)] += 0.1
                        else:
                            sfc_info[1 + self.num_vnf_types + VNF_TYPES.index(vnf)] += 0.1
                    break
            
            state2.extend(sfc_info)
        
        state3 = []
        for sfc_type in SFC_TYPES:
            pending_count = sum(1 for r in self.sfc_requests if r['type'] == sfc_type and r['status'] == 'pending')
            
            min_remaining = float('inf')
            total_bw = 0
            vnf_waiting = {vnf: 0 for vnf in VNF_TYPES}
            
            for req in self.sfc_requests:
                if req['type'] == sfc_type and req['status'] == 'pending':
                    remaining = req['delay_tolerance'] - (self.current_time - req['creation_time'])
                    min_remaining = min(min_remaining, remaining)
                    total_bw = req['bandwidth']
                    
                    for vnf_idx, vnf in enumerate(req['chain']):
                        if vnf_idx >= len(req['allocated_vnfs']):
                            vnf_waiting[vnf] += 1
            
            if min_remaining == float('inf'):
                min_remaining = 0
            
            info = [SFC_TYPES.index(sfc_type) / len(SFC_TYPES)]
            info.append(pending_count / 50)
            info.append(min(1.0, min_remaining / 100))
            info.append(total_bw / 100)
            
            for vnf in VNF_TYPES:
                info.append(vnf_waiting[vnf] / 20)
            
            state3.extend(info)
        
        return {
            'state1': np.array(state1, dtype=np.float32),
            'state2': np.array(state2, dtype=np.float32),
            'state3': np.array(state3, dtype=np.float32)
        }
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == 2 * self.num_vnf_types:
            reward = 0
        elif action < self.num_vnf_types:
            reward = self._allocate_vnf(action)
        else:
            reward = self._uninstall_vnf(action - self.num_vnf_types)
        
        completion_reward = self._check_sfc_completion()
        reward += completion_reward
        
        self._update_dc_priority()
        
        pending_count = len([r for r in self.sfc_requests if r['status'] == 'pending'])
        if pending_count == 0:
            done = True
        
        self.current_time += TRAINING_CONFIG['action_inference_time']
        
        return self._get_state(), reward, done, False, {'pending': pending_count}
    
    def _allocate_vnf(self, vnf_idx):
        vnf_type = VNF_TYPES[vnf_idx]
        dc = self.dcs[self.current_dc_idx]
        requirements = VNF_REQUIREMENTS[vnf_type]
        
        pending_vnfs = self._get_pending_vnfs(vnf_type)
        if not pending_vnfs:
            return REWARD_CONFIG['invalid_action']
        
        if dc['cpu'] - dc['cpu_used'] < requirements['cpu']:
            dc['installed_vnfs'][vnf_type] += 1
            dc['cpu_used'] += requirements['cpu']
            dc['ram_used'] += requirements['ram']
            dc['storage_used'] += requirements['storage']
        
        if dc['installed_vnfs'][vnf_type] <= dc['allocated_vnfs'][vnf_type]:
            return REWARD_CONFIG['invalid_action']
        
        selected_vnf = self._select_vnf_with_priority(pending_vnfs, vnf_type)
        
        selected_vnf['req']['allocated_vnfs'].append({
            'vnf': vnf_type,
            'dc_idx': self.current_dc_idx,
            'alloc_time': self.current_time
        })
        
        dc['allocated_vnfs'][vnf_type] += 1
        
        return 0
    
    def _uninstall_vnf(self, vnf_idx):
        vnf_type = VNF_TYPES[vnf_idx]
        dc = self.dcs[self.current_dc_idx]
        
        if dc['installed_vnfs'][vnf_type] <= dc['allocated_vnfs'][vnf_type]:
            return REWARD_CONFIG['uninstall_required']
        
        if dc['installed_vnfs'][vnf_type] > 0:
            dc['installed_vnfs'][vnf_type] -= 1
            requirements = VNF_REQUIREMENTS[vnf_type]
            dc['cpu_used'] -= requirements['cpu']
            dc['ram_used'] -= requirements['ram']
            dc['storage_used'] -= requirements['storage']
            return 0
        
        return REWARD_CONFIG['invalid_action']
    
    def _get_pending_vnfs(self, vnf_type):
        pending = []
        for req in self.sfc_requests:
            if req['status'] != 'pending':
                continue
            
            next_vnf_idx = len(req['allocated_vnfs'])
            if next_vnf_idx < len(req['chain']) and req['chain'][next_vnf_idx] == vnf_type:
                pending.append({'req': req, 'vnf_idx': next_vnf_idx})
        
        return pending
    
    def _select_vnf_with_priority(self, pending_vnfs, vnf_type):
        max_priority = -float('inf')
        selected = None
        
        for item in pending_vnfs:
            req = item['req']
            elapsed = self.current_time - req['creation_time']
            remaining = req['delay_tolerance'] - elapsed
            
            p1 = elapsed
            p2 = self._calculate_dc_priority(req)
            p3 = 0
            if remaining < PRIORITY_CONFIG['urgency_threshold']:
                p3 = PRIORITY_CONFIG['urgency_constant'] / (remaining + 0.001)
            
            priority = p1 + p2 + p3
            
            if priority > max_priority:
                max_priority = priority
                selected = item
        
        return selected
    
    def _calculate_dc_priority(self, req):
        priority = 0
        dc_idx = self.current_dc_idx
        
        for alloc in req['allocated_vnfs']:
            if alloc['dc_idx'] == dc_idx:
                priority += 10
            else:
                priority -= 5
        
        return priority
    
    def _update_dc_priority(self):
        self.current_dc_idx = (self.current_dc_idx + 1) % self.num_dcs
    
    def _check_sfc_completion(self):
        total_reward = 0
        
        for req in self.sfc_requests:
            if req['status'] != 'pending':
                continue
            
            elapsed = self.current_time - req['creation_time']
            
            if len(req['allocated_vnfs']) == len(req['chain']):
                total_delay = self._calculate_total_delay(req)
                
                if total_delay <= req['delay_tolerance']:
                    req['status'] = 'accepted'
                    self.total_accepted += 1
                    total_reward += REWARD_CONFIG['sfc_satisfied']
                else:
                    req['status'] = 'dropped'
                    self.total_dropped += 1
                    self._release_resources(req)
                    total_reward += REWARD_CONFIG['sfc_dropped']
            
            elif elapsed > req['delay_tolerance']:
                req['status'] = 'dropped'
                self.total_dropped += 1
                self._release_resources(req)
                total_reward += REWARD_CONFIG['sfc_dropped']
        
        return total_reward
    
    def _calculate_total_delay(self, req):
        prop_delay = 0
        proc_delay = 0
        
        for i in range(len(req['allocated_vnfs']) - 1):
            dc1 = req['allocated_vnfs'][i]['dc_idx']
            dc2 = req['allocated_vnfs'][i+1]['dc_idx']
            prop_delay += calculate_propagation_delay(self.distance_matrix[dc1][dc2])
        
        for alloc in req['allocated_vnfs']:
            vnf = alloc['vnf']
            proc_delay += VNF_REQUIREMENTS[vnf]['proc_time']
        
        return prop_delay + proc_delay
    
    def _release_resources(self, req):
        for alloc in req['allocated_vnfs']:
            dc = self.dcs[alloc['dc_idx']]
            dc['allocated_vnfs'][alloc['vnf']] -= 1
    
    def add_requests(self, requests):
        for req in requests:
            req['creation_time'] = self.current_time
        self.sfc_requests.extend(requests)
        self.total_generated += len(requests)
    
    def get_acceptance_ratio(self):
        if self.total_generated == 0:
            return 0
        return self.total_accepted / self.total_generated