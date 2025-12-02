import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
sys.path.append('..')
from config import *
from utils import *

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
        
        return self._get_observation(), {}
    
    def _generate_sfc_requests(self):
        num_types = np.random.randint(1, len(self.sfc_types) + 1)
        selected_types = np.random.choice(self.sfc_types, num_types, replace=False)
        
        total_created = 0
        
        for sfc_type in selected_types:
            config = SFC_TYPES[sfc_type]
            raw_bundle = np.random.randint(*config['bundle'])
            # scale traffic so we can run lighter/faster simulations during debugging
            bundle_size = max(1, int(raw_bundle * SIM_CONFIG.get('traffic_scale', 1.0)))
            # avoid creating more pending SFCs than allowed
            max_pending = SIM_CONFIG.get('max_pending_sfcs', 2000)
            remaining_slots = max_pending - (len(self.pending_sfcs) + total_created)
            if remaining_slots <= 0:
                break
            if bundle_size > remaining_slots:
                bundle_size = remaining_slots
            
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

        if total_created and SIM_CONFIG.get('debug', False):
            print(f"[ENV DEBUG] Created {total_created} SFCs (pending total now {len(self.pending_sfcs)})")
    
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
        
        state2 = []
        for sfc_type in self.sfc_types:
            sfc_info = [0] * (1 + 2 * len(VNF_LIST))
            
            for sfc in self.active_sfcs:
                if sfc['type'] == sfc_type and current_dc['id'] in sfc['allocated_dcs']:
                    sfc_info[0] = 1
                    for vnf in sfc['allocated_vnfs']:
                        if vnf in VNF_LIST:
                            idx = VNF_LIST.index(vnf)
                            sfc_info[1 + idx] = 1
                    
                    remaining_vnfs = [v for v in sfc['vnfs'] if v not in sfc['allocated_vnfs']]
                    for vnf in remaining_vnfs:
                        if vnf in VNF_LIST:
                            idx = VNF_LIST.index(vnf)
                            sfc_info[1 + len(VNF_LIST) + idx] = 1
                    break
            
            state2.extend(sfc_info)
        
        state3 = []
        for sfc_type in self.sfc_types:
            type_sfcs = [s for s in self.pending_sfcs if s['type'] == sfc_type]
            
            if type_sfcs:
                count = len(type_sfcs)
                min_remaining = min([s['delay'] - (self.current_time - s['created_time']) 
                                    for s in type_sfcs])
                bw = type_sfcs[0]['bw']
                
                vnf_counts = [0] * len(VNF_LIST)
                for sfc in type_sfcs:
                    remaining = [v for v in sfc['vnfs'] if v not in sfc['allocated_vnfs']]
                    for vnf in remaining:
                        if vnf in VNF_LIST:
                            vnf_counts[VNF_LIST.index(vnf)] += 1
                
                state3.extend([1, count, max(0, min_remaining), bw] + vnf_counts)
            else:
                state3.extend([0] * (4 + len(VNF_LIST)))
        
        return {
            'state1': np.array(state1, dtype=np.float32),
            'state2': np.array(state2, dtype=np.float32),
            'state3': np.array(state3, dtype=np.float32)
        }
    
    def step(self, action):
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
        
        self._update_sfcs()
        
        self.current_dc_idx = (self.current_dc_idx + 1) % max(1, len(self.dc_priority_list))
        
        if self.current_dc_idx == 0:
            self.current_time += DRL_CONFIG['action_inference_time']
            self._set_dc_priority()
        
        if len(self.pending_sfcs) == 0 and len(self.active_sfcs) == 0:
            done = True
        
        obs = self._get_observation()
        
        total_requests = len(self.satisfied_sfcs) + len(self.dropped_sfcs)
        acceptance_ratio = len(self.satisfied_sfcs) / max(1, total_requests)
        
        avg_delay = 0
        if self.satisfied_sfcs:
            total_delay = 0
            for sfc in self.satisfied_sfcs:
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
                total_delay += path_delay + proc_delay
            
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
        sfc['allocated_vnfs'].append(vnf_type)
        sfc['allocated_dcs'].append(dc['id'])
        sfc['processing_times'].append(VNF_SPECS[vnf_type]['proc_time'])
        
        if sfc['id'] not in self.sfc_allocated_dcs:
            self.sfc_allocated_dcs[sfc['id']] = []
        self.sfc_allocated_dcs[sfc['id']].append(dc['id'])
        
        if sfc in self.pending_sfcs:
            self.pending_sfcs.remove(sfc)
            self.active_sfcs.append(sfc)
        
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
                if SIM_CONFIG.get('debug', False):
                    print(f"[ENV DEBUG] SFC {sfc['id']} ({sfc['type']}) dropped: elapsed={self.current_time - sfc['created_time']:.2f}, delay_budget={sfc['delay']}")
                return REWARD_CONFIG['sfc_dropped']
        
        return REWARD_CONFIG['default']
    
    def _check_sfc_completion(self, sfc):
        path_delay = 0
        proc_delay = 0
        waiting_delay = 0
        
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
                else:
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
        
        return total_future_delay <= remaining_budget
    
    def _update_sfcs(self):
        to_drop = []
        for sfc in self.pending_sfcs + self.active_sfcs:
            elapsed = self.current_time - sfc['created_time']
            if elapsed > sfc['delay']:
                to_drop.append(sfc)
        
        for sfc in to_drop:
            if sfc in self.pending_sfcs:
                self.pending_sfcs.remove(sfc)
            elif sfc in self.active_sfcs:
                self.active_sfcs.remove(sfc)
                self._release_resources(sfc)
            self.dropped_sfcs.append(sfc)
    
    def _release_resources(self, sfc):
        for i, vnf_type in enumerate(sfc['allocated_vnfs']):
            dc_id = sfc['allocated_dcs'][i]
            dc = self.dcs[dc_id]
            dc['allocated_vnfs'][vnf_type] -= 1