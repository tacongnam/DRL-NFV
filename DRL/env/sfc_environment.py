import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import time
sys.path.append('..')
from config import *
from utils import *

# Timing metrics
TIMING_STATS = {
    'pathfinding': 0,
    'pathfinding_calls': 0,
    'update_sfcs': 0,
    'check_completion': 0,
    'get_observation': 0,
    'step_total': 0
}

class SFCEnvironment(gym.Env):
    def __init__(self, num_dcs=4):
        super(SFCEnvironment, self).__init__()
        
        self.num_dcs = num_dcs
        self.vnf_types = VNF_LIST
        self.sfc_types = list(SFC_TYPES.keys())
        
        self.action_space = spaces.Discrete(2 * len(VNF_LIST) + 1)
        
        obs_dim1 = 2 * len(VNF_LIST) + 2
        obs_dim2 = len(self.sfc_types) * (1 + 2 * len(VNF_LIST))
        obs_dim3 = len(self.sfc_types) * (4 + len(VNF_LIST))
        
        self.observation_space = spaces.Dict({
            'state1': spaces.Box(low=0, high=np.inf, shape=(obs_dim1,), dtype=np.float32),
            'state2': spaces.Box(low=0, high=np.inf, shape=(obs_dim2,), dtype=np.float32),
            'state3': spaces.Box(low=0, high=np.inf, shape=(obs_dim3,), dtype=np.float32)
        })
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.network = create_network_topology(self.num_dcs)
        
        self.dcs = []
        for i in range(self.num_dcs):
            cpu = np.random.randint(*DC_CONFIG['cpu_range'])
            self.dcs.append({
                'id': i,
                'cpu': cpu,
                'ram': DC_CONFIG['ram'],
                'storage': DC_CONFIG['storage'],
                'available_cpu': cpu,
                'available_ram': DC_CONFIG['ram'],
                'available_storage': DC_CONFIG['storage'],
                'installed_vnfs': {vnf: 0 for vnf in VNF_LIST},
                'allocated_vnfs': {vnf: 0 for vnf in VNF_LIST}
            })
        
        self.pending_sfcs = []
        self.active_sfcs = []
        self.satisfied_sfcs = []
        self.dropped_sfcs = []
        self.current_time = 0
        self.current_dc_idx = 0
        self.dc_priority_list = list(range(self.num_dcs))
        self.sfc_allocated_dcs = {}
        
        # Cache SFCs by next VNF needed for faster lookup
        self._pending_by_next_vnf = {vnf: [] for vnf in VNF_LIST}
        self._active_by_next_vnf = {vnf: [] for vnf in VNF_LIST}
        
        return self._get_observation(), {}
    
    def _generate_sfc_requests(self):
        num_types = np.random.randint(1, len(self.sfc_types) + 1)
        selected_types = np.random.choice(self.sfc_types, num_types, replace=False)
        
        # Limit pending SFCs to avoid queue explosion
        # If queue is too large, skip request generation this round
        MAX_PENDING = 50  # Hard limit
        if len(self.pending_sfcs) >= MAX_PENDING:
            return
        
        total_created = 0
        
        for sfc_type in selected_types:
            config = SFC_TYPES[sfc_type]
            bundle_size = np.random.randint(*config['bundle'])
            
            # Limit bundle to avoid creating too many at once
            bundle_size = min(bundle_size, MAX_PENDING - len(self.pending_sfcs))
            if bundle_size <= 0:
                break
            
            for _ in range(bundle_size):
                source = np.random.randint(0, self.num_dcs)
                dest = np.random.randint(0, self.num_dcs)
                while dest == source:
                    dest = np.random.randint(0, self.num_dcs)
                
                # Ensure bandwidth is a scalar. If config['bw'] is a range (tuple/list), sample a value.
                bw_config = config.get('bw')
                if isinstance(bw_config, (list, tuple)) and len(bw_config) == 2:
                    # sample uniformly within the provided range
                    bw_value = float(np.random.uniform(bw_config[0], bw_config[1]))
                else:
                    bw_value = float(bw_config)

                sfc = {
                    'id': len(self.pending_sfcs) + len(self.active_sfcs) + len(self.satisfied_sfcs) + len(self.dropped_sfcs),
                    'type': sfc_type,
                    'vnfs': config['vnfs'].copy(),
                    'bw': bw_value,
                    'delay': config['delay'],
                    'source': source,
                    'dest': dest,
                    'created_time': self.current_time,
                    'allocated_vnfs': [],
                    'allocated_dcs': [],
                    'processing_times': []
                }
                self.pending_sfcs.append(sfc)
                total_created += 1
    
    def _set_dc_priority(self):
        priorities = []
        min_delay_sfc = None
        min_delay = float('inf')
        
        for sfc in self.pending_sfcs:
            remaining = sfc['delay'] - (self.current_time - sfc['created_time'])
            if remaining < min_delay:
                min_delay = remaining
                min_delay_sfc = sfc
        
        if min_delay_sfc is None:
            self.dc_priority_list = list(range(self.num_dcs))
            return
        
        source = min_delay_sfc['source']
        dest = min_delay_sfc['dest']
        path = get_shortest_path_with_bw(self.network, source, dest, min_delay_sfc['bw'])
        
        if path:
            priority_dcs = path.copy()
            other_dcs = [i for i in range(self.num_dcs) if i not in path]
            self.dc_priority_list = priority_dcs + other_dcs
        else:
            self.dc_priority_list = list(range(self.num_dcs))
    
    def _get_observation(self):
        if len(self.dc_priority_list) == 0:
            self.dc_priority_list = list(range(self.num_dcs))
        
        if self.current_dc_idx >= len(self.dc_priority_list):
            self.current_dc_idx = 0
        
        current_dc = self.dcs[self.dc_priority_list[self.current_dc_idx]]
        
        state1 = []
        for vnf in VNF_LIST:
            state1.append(current_dc['installed_vnfs'][vnf])
            state1.append(current_dc['installed_vnfs'][vnf] - current_dc['allocated_vnfs'][vnf])
        state1.extend([current_dc['available_storage'], current_dc['available_cpu']])
        
        # Pre-build indices for faster lookups
        vnf_to_idx = {vnf: idx for idx, vnf in enumerate(VNF_LIST)}
        current_dc_id = current_dc['id']
        
        # Group active SFCs by type for faster lookup in state2
        active_by_type = {sfc_type: None for sfc_type in self.sfc_types}
        for sfc in self.active_sfcs:
            if current_dc_id in sfc['allocated_dcs'] and active_by_type[sfc['type']] is None:
                active_by_type[sfc['type']] = sfc
        
        state2 = []
        for sfc_type in self.sfc_types:
            sfc_info = [0] * (1 + 2 * len(VNF_LIST))
            
            sfc = active_by_type[sfc_type]
            if sfc:
                sfc_info[0] = 1
                for vnf in sfc['allocated_vnfs']:
                    if vnf in vnf_to_idx:
                        sfc_info[1 + vnf_to_idx[vnf]] = 1
                
                remaining_vnfs = [v for v in sfc['vnfs'] if v not in sfc['allocated_vnfs']]
                for vnf in remaining_vnfs:
                    if vnf in vnf_to_idx:
                        sfc_info[1 + len(VNF_LIST) + vnf_to_idx[vnf]] = 1
            
            state2.extend(sfc_info)
        
        # Group pending SFCs by type for state3
        pending_by_type = {sfc_type: [] for sfc_type in self.sfc_types}
        for sfc in self.pending_sfcs:
            pending_by_type[sfc['type']].append(sfc)
        
        state3 = []
        for sfc_type in self.sfc_types:
            type_sfcs = pending_by_type[sfc_type]
            
            if type_sfcs:
                count = len(type_sfcs)
                min_remaining = min([s['delay'] - (self.current_time - s['created_time']) 
                                    for s in type_sfcs])
                bw = type_sfcs[0]['bw']
                
                vnf_counts = [0] * len(VNF_LIST)
                for sfc in type_sfcs:
                    remaining = [v for v in sfc['vnfs'] if v not in sfc['allocated_vnfs']]
                    for vnf in remaining:
                        if vnf in vnf_to_idx:
                            vnf_counts[vnf_to_idx[vnf]] += 1
                
                state3.extend([1, count, max(0, min_remaining), bw] + vnf_counts)
            else:
                state3.extend([0] * (4 + len(VNF_LIST)))
        
        return {
            'state1': np.array(state1, dtype=np.float32),
            'state2': np.array(state2, dtype=np.float32),
            'state3': np.array(state3, dtype=np.float32)
        }
    
    def step(self, action):
        step_start = time.time()
        reward = REWARD_CONFIG['default']
        done = False
        
        if len(self.dc_priority_list) == 0:
            self.dc_priority_list = list(range(self.num_dcs))
        
        if self.current_time % DRL_CONFIG['request_interval'] == 0 and self.current_dc_idx == 0:
            self._generate_sfc_requests()
        
        if action == 2 * len(VNF_LIST):
            pass
        elif action < len(VNF_LIST):
            reward = self._uninstall_vnf(action)
        else:
            reward = self._allocate_vnf(action - len(VNF_LIST))
        
        update_start = time.time()
        self._update_sfcs()
        TIMING_STATS['update_sfcs'] += time.time() - update_start
        
        self.current_dc_idx = (self.current_dc_idx + 1) % max(1, len(self.dc_priority_list))
        
        if self.current_dc_idx == 0:
            self.current_time += DRL_CONFIG['action_inference_time']
            self._set_dc_priority()
        
        if len(self.pending_sfcs) == 0 and len(self.active_sfcs) == 0:
            done = True
        
        obs_start = time.time()
        obs = self._get_observation()
        TIMING_STATS['get_observation'] += time.time() - obs_start
        
        total_requests = len(self.satisfied_sfcs) + len(self.dropped_sfcs)
        acceptance_ratio = len(self.satisfied_sfcs) / max(1, total_requests)
        
        avg_delay = 0
        if self.satisfied_sfcs:
            total_delay = 0
            for sfc in self.satisfied_sfcs:
                # Use cached delay if available, otherwise calculate
                if 'cached_delay' not in sfc:
                    path_delay = 0
                    unique_dcs = []
                    for dc_id in sfc['allocated_dcs']:
                        if not unique_dcs or dc_id != unique_dcs[-1]:
                            unique_dcs.append(dc_id)
                    
                    if len(unique_dcs) > 1:
                        for i in range(len(unique_dcs) - 1):
                            path = get_shortest_path_with_bw(self.network, unique_dcs[i], 
                                                             unique_dcs[i+1], sfc['bw'])
                            if path:
                                path_delay += calculate_propagation_delay(self.network, path)
                    
                    proc_delay = sum(sfc['processing_times'])
                    sfc['cached_delay'] = path_delay + proc_delay
                
                total_delay += sfc['cached_delay']
            
            avg_delay = total_delay / len(self.satisfied_sfcs)
        
        resource_util = 0
        total_cpu = sum(dc['cpu'] for dc in self.dcs)
        if total_cpu > 0:
            used_cpu = sum(dc['cpu'] - dc['available_cpu'] for dc in self.dcs)
            resource_util = used_cpu / total_cpu
        
        info = {
            'satisfied': len(self.satisfied_sfcs),
            'dropped': len(self.dropped_sfcs),
            'acceptance_ratio': acceptance_ratio,
            'avg_delay': avg_delay,
            'resource_util': resource_util
        }
        
        TIMING_STATS['step_total'] += time.time() - step_start
        
        return obs, reward, done, False, info
    
    def _uninstall_vnf(self, vnf_idx):
        vnf_type = VNF_LIST[vnf_idx]
        current_dc = self.dcs[self.dc_priority_list[self.current_dc_idx]]
        
        has_pending = False
        for sfc in self.pending_sfcs + self.active_sfcs:
            remaining_vnfs = [v for v in sfc['vnfs'] if v not in sfc['allocated_vnfs']]
            if vnf_type in remaining_vnfs:
                has_pending = True
                break
        
        idle_count = current_dc['installed_vnfs'][vnf_type] - current_dc['allocated_vnfs'][vnf_type]
        
        if idle_count > 0 and not has_pending:
            current_dc['installed_vnfs'][vnf_type] -= 1
            specs = VNF_SPECS[vnf_type]
            current_dc['available_cpu'] += specs['cpu']
            current_dc['available_ram'] += specs['ram']
            current_dc['available_storage'] += specs['storage']
            return REWARD_CONFIG['default']
        elif has_pending:
            return REWARD_CONFIG['uninstall_required']
        else:
            return REWARD_CONFIG['invalid_action']
    
    def _allocate_vnf(self, vnf_idx):
        vnf_type = VNF_LIST[vnf_idx]
        current_dc = self.dcs[self.dc_priority_list[self.current_dc_idx]]
        
        # Use cached lists if available, otherwise build them
        waiting_vnfs = (self._pending_by_next_vnf.get(vnf_type, []) + 
                       self._active_by_next_vnf.get(vnf_type, []))
        
        if not waiting_vnfs:
            # Fall back to searching through all SFCs
            waiting_vnfs = []
            for sfc in self.pending_sfcs + self.active_sfcs:
                if len(sfc['allocated_vnfs']) < len(sfc['vnfs']):
                    next_vnf_idx = len(sfc['allocated_vnfs'])
                    if next_vnf_idx < len(sfc['vnfs']) and sfc['vnfs'][next_vnf_idx] == vnf_type:
                        waiting_vnfs.append(sfc)
        
        if not waiting_vnfs:
            return REWARD_CONFIG['invalid_action']
        
        can_allocate = False
        if current_dc['installed_vnfs'][vnf_type] > current_dc['allocated_vnfs'][vnf_type]:
            can_allocate = True
        elif check_resource_availability(current_dc, vnf_type, VNF_SPECS):
            can_allocate = True
        
        if not can_allocate:
            return REWARD_CONFIG['invalid_action']
        
        selected_sfc = self._select_vnf_by_priority(waiting_vnfs, current_dc['id'])
        
        if selected_sfc:
            return self._perform_allocation(selected_sfc, vnf_type, current_dc)
        
        return REWARD_CONFIG['invalid_action']
    
    def _select_vnf_by_priority(self, waiting_vnfs, dc_id):
        best_sfc = None
        best_priority = float('-inf')
        
        for sfc in waiting_vnfs:
            elapsed = self.current_time - sfc['created_time']
            
            p1 = calculate_priority_p1(elapsed, sfc['delay'])
            p2 = calculate_priority_p2({'sfc_id': sfc['id']}, dc_id, self.sfc_allocated_dcs)
            p3 = calculate_priority_p3(elapsed, sfc['delay'])
            
            priority = p1 + p2 + p3
            
            if priority > best_priority:
                best_priority = priority
                best_sfc = sfc
        
        return best_sfc
    
    def _perform_allocation(self, sfc, vnf_type, dc):
        if dc['installed_vnfs'][vnf_type] <= dc['allocated_vnfs'][vnf_type]:
            specs = VNF_SPECS[vnf_type]
            if check_resource_availability(dc, vnf_type, VNF_SPECS):
                dc['installed_vnfs'][vnf_type] += 1
                dc['available_cpu'] -= specs['cpu']
                dc['available_ram'] -= specs['ram']
                dc['available_storage'] -= specs['storage']
            else:
                return REWARD_CONFIG['invalid_action']
        
        dc['allocated_vnfs'][vnf_type] += 1
        
        # Update cache: remove from current VNF list
        next_vnf_idx = len(sfc['allocated_vnfs'])
        if sfc in self.pending_sfcs:
            if sfc in self._pending_by_next_vnf.get(vnf_type, []):
                self._pending_by_next_vnf[vnf_type].remove(sfc)
        else:
            if sfc in self._active_by_next_vnf.get(vnf_type, []):
                self._active_by_next_vnf[vnf_type].remove(sfc)
        
        sfc['allocated_vnfs'].append(vnf_type)
        sfc['allocated_dcs'].append(dc['id'])
        sfc['processing_times'].append(VNF_SPECS[vnf_type]['proc_time'])
        
        if sfc['id'] not in self.sfc_allocated_dcs:
            self.sfc_allocated_dcs[sfc['id']] = []
        self.sfc_allocated_dcs[sfc['id']].append(dc['id'])
        
        if sfc in self.pending_sfcs:
            self.pending_sfcs.remove(sfc)
            self.active_sfcs.append(sfc)
            # Update cache: move to active
            if next_vnf_idx < len(sfc['vnfs']):
                next_vnf = sfc['vnfs'][next_vnf_idx]
                if sfc not in self._active_by_next_vnf.get(next_vnf, []):
                    self._active_by_next_vnf[next_vnf].append(sfc)
        else:
            # Already active, just update the next VNF cache
            if next_vnf_idx < len(sfc['vnfs']):
                next_vnf = sfc['vnfs'][next_vnf_idx]
                if sfc not in self._active_by_next_vnf.get(next_vnf, []):
                    self._active_by_next_vnf[next_vnf].append(sfc)
        
        if len(sfc['allocated_vnfs']) == len(sfc['vnfs']):
            if self._check_sfc_completion(sfc):
                if sfc in self.active_sfcs:
                    self.active_sfcs.remove(sfc)
                self.satisfied_sfcs.append(sfc)
                self._release_resources(sfc)
                return REWARD_CONFIG['sfc_satisfied']
            else:
                if sfc in self.active_sfcs:
                    self.active_sfcs.remove(sfc)
                self.dropped_sfcs.append(sfc)
                self._release_resources(sfc)
                return REWARD_CONFIG['sfc_dropped']
        
        return REWARD_CONFIG['default']
    
    def _check_sfc_completion(self, sfc):
        check_start = time.time()
        path_delay = 0
        proc_delay = 0
        waiting_delay = 0
        
        unique_dcs = []
        for dc_id in sfc['allocated_dcs']:
            if not unique_dcs or dc_id != unique_dcs[-1]:
                unique_dcs.append(dc_id)
        
        if len(unique_dcs) > 1:
            for i in range(len(unique_dcs) - 1):
                path_start = time.time()
                path = get_shortest_path_with_bw(self.network, unique_dcs[i], 
                                                 unique_dcs[i+1], sfc['bw'])
                TIMING_STATS['pathfinding'] += time.time() - path_start
                TIMING_STATS['pathfinding_calls'] += 1
                
                if path:
                    path_delay += calculate_propagation_delay(self.network, path)
                else:
                    TIMING_STATS['check_completion'] += time.time() - check_start
                    return False
        
        proc_delay = sum(sfc['processing_times'])
        
        for i, vnf_type in enumerate(sfc['allocated_vnfs']):
            dc_id = sfc['allocated_dcs'][i]
            dc = self.dcs[dc_id]
            
            allocated_count = sum(1 for j, v in enumerate(sfc['allocated_vnfs'][:i]) 
                                 if v == vnf_type and sfc['allocated_dcs'][j] == dc_id)
            
            queue_length = allocated_count
            waiting_delay += queue_length * VNF_SPECS[vnf_type]['proc_time']
        
        # Only check future delays, not elapsed time
        total_future_delay = path_delay + proc_delay + waiting_delay
        elapsed = self.current_time - sfc['created_time']
        remaining_budget = sfc['delay'] - elapsed
        
        TIMING_STATS['check_completion'] += time.time() - check_start
        return total_future_delay <= remaining_budget
    
    def _update_sfcs(self):
        # Only check pending SFCs since active ones are already processed
        to_drop = []
        
        # Check pending SFCs
        i = 0
        while i < len(self.pending_sfcs):
            sfc = self.pending_sfcs[i]
            elapsed = self.current_time - sfc['created_time']
            if elapsed > sfc['delay']:
                to_drop.append((sfc, 'pending'))
            i += 1
        
        # Check active SFCs
        i = 0
        while i < len(self.active_sfcs):
            sfc = self.active_sfcs[i]
            elapsed = self.current_time - sfc['created_time']
            if elapsed > sfc['delay']:
                to_drop.append((sfc, 'active'))
            i += 1
        
        # Remove dropped SFCs and clean caches
        for sfc, source in to_drop:
            # Clean from cache
            if len(sfc['allocated_vnfs']) < len(sfc['vnfs']):
                next_vnf_idx = len(sfc['allocated_vnfs'])
                next_vnf = sfc['vnfs'][next_vnf_idx]
                if source == 'pending' and sfc in self._pending_by_next_vnf.get(next_vnf, []):
                    self._pending_by_next_vnf[next_vnf].remove(sfc)
                elif source == 'active' and sfc in self._active_by_next_vnf.get(next_vnf, []):
                    self._active_by_next_vnf[next_vnf].remove(sfc)
            
            # Remove from state
            if source == 'pending':
                self.pending_sfcs.remove(sfc)
            else:  # active
                self.active_sfcs.remove(sfc)
                self._release_resources(sfc)
            self.dropped_sfcs.append(sfc)
    
    def _release_resources(self, sfc):
        for i, vnf_type in enumerate(sfc['allocated_vnfs']):
            dc_id = sfc['allocated_dcs'][i]
            dc = self.dcs[dc_id]
            dc['allocated_vnfs'][vnf_type] -= 1