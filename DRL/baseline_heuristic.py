import numpy as np
from env.sfc_environment import SFCEnvironment
from config import *
from utils import *

class BaselineHeuristic:
    def __init__(self, env):
        self.env = env
    
    def run(self, max_steps=200):
        self.env.reset()
        step = 0
        
        while step < max_steps:
            if len(self.env.pending_sfcs) == 0 and len(self.env.active_sfcs) == 0:
                break
            
            if self.env.current_time % DRL_CONFIG['request_interval'] == 0:
                self.env._generate_sfc_requests()
            
            if self.env.pending_sfcs or self.env.active_sfcs:
                self._process_sfcs()
            
            self.env._update_sfcs()
            self.env.current_time += DRL_CONFIG['action_inference_time']
            step += 1
        
        total = len(self.env.satisfied_sfcs) + len(self.env.dropped_sfcs)
        acceptance_ratio = len(self.env.satisfied_sfcs) / max(1, total)
        
        avg_delay = 0
        if self.env.satisfied_sfcs:
            total_delay = 0
            for sfc in self.env.satisfied_sfcs:
                path_delay = 0
                unique_dcs = []
                for dc_id in sfc['allocated_dcs']:
                    if not unique_dcs or dc_id != unique_dcs[-1]:
                        unique_dcs.append(dc_id)
                
                if len(unique_dcs) > 1:
                    for i in range(len(unique_dcs) - 1):
                        path = get_shortest_path_with_bw(self.env.network, unique_dcs[i], 
                                                         unique_dcs[i+1], sfc['bw'])
                        if path:
                            path_delay += calculate_propagation_delay(self.env.network, path)
                
                proc_delay = sum(sfc['processing_times'])
                total_delay += path_delay + proc_delay
            
            avg_delay = total_delay / len(self.env.satisfied_sfcs)
        
        resource_util = 0
        total_cpu = sum(dc['cpu'] for dc in self.env.dcs)
        if total_cpu > 0:
            used_cpu = sum(dc['cpu'] - dc['available_cpu'] for dc in self.env.dcs)
            resource_util = used_cpu / total_cpu
        
        return {
            'acceptance_ratio': acceptance_ratio,
            'avg_delay': avg_delay,
            'resource_util': resource_util,
            'satisfied': len(self.env.satisfied_sfcs),
            'dropped': len(self.env.dropped_sfcs)
        }
    
    def _process_sfcs(self):
        all_sfcs = sorted(
            self.env.pending_sfcs + self.env.active_sfcs,
            key=lambda s: s['delay'] - (self.env.current_time - s['created_time'])
        )
        
        for sfc in all_sfcs:
            if len(sfc['allocated_vnfs']) >= len(sfc['vnfs']):
                continue
            
            vnf_idx = len(sfc['allocated_vnfs'])
            vnf_type = sfc['vnfs'][vnf_idx]
            
            path = get_shortest_path_with_bw(
                self.env.network, sfc['source'], sfc['dest'], sfc['bw']
            )
            
            if not path:
                continue
            
            best_dc = None
            best_score = float('-inf')
            
            for dc_id in path:
                dc = self.env.dcs[dc_id]
                
                can_allocate = False
                if dc['installed_vnfs'][vnf_type] > dc['allocated_vnfs'][vnf_type]:
                    can_allocate = True
                elif check_resource_availability(dc, vnf_type, VNF_SPECS):
                    can_allocate = True
                
                if can_allocate:
                    score = 0
                    if dc_id in sfc['allocated_dcs']:
                        score += 100
                    
                    score += dc['available_cpu'] / dc['cpu']
                    
                    if score > best_score:
                        best_score = score
                        best_dc = dc
            
            if best_dc:
                self._allocate_vnf_to_dc(sfc, vnf_type, best_dc)
    
    def _allocate_vnf_to_dc(self, sfc, vnf_type, dc):
        if dc['installed_vnfs'][vnf_type] <= dc['allocated_vnfs'][vnf_type]:
            specs = VNF_SPECS[vnf_type]
            if check_resource_availability(dc, vnf_type, VNF_SPECS):
                dc['installed_vnfs'][vnf_type] += 1
                dc['available_cpu'] -= specs['cpu']
                dc['available_ram'] -= specs['ram']
                dc['available_storage'] -= specs['storage']
            else:
                return
        
        dc['allocated_vnfs'][vnf_type] += 1
        sfc['allocated_vnfs'].append(vnf_type)
        sfc['allocated_dcs'].append(dc['id'])
        sfc['processing_times'].append(VNF_SPECS[vnf_type]['proc_time'])
        
        if sfc in self.env.pending_sfcs:
            self.env.pending_sfcs.remove(sfc)
            self.env.active_sfcs.append(sfc)
        
        if len(sfc['allocated_vnfs']) == len(sfc['vnfs']):
            if self.env._check_sfc_completion(sfc):
                if sfc in self.env.active_sfcs:
                    self.env.active_sfcs.remove(sfc)
                self.env.satisfied_sfcs.append(sfc)
                self.env._release_resources(sfc)
            else:
                if sfc in self.env.active_sfcs:
                    self.env.active_sfcs.remove(sfc)
                self.env.dropped_sfcs.append(sfc)
                self.env._release_resources(sfc)

def compare_with_baseline(agent, num_dcs=4, num_tests=5):
    print("\n" + "="*60)
    print("Comparing DRL Agent vs Baseline Heuristic")
    print("="*60)
    
    drl_results = {
        'acceptance_ratio': [],
        'avg_delay': [],
        'resource_util': []
    }
    
    baseline_results = {
        'acceptance_ratio': [],
        'avg_delay': [],
        'resource_util': []
    }
    
    for test in range(num_tests):
        print(f"\nTest {test + 1}/{num_tests}")
        
        env_drl = SFCEnvironment(num_dcs=num_dcs)
        state, _ = env_drl.reset()
        
        done = False
        step_count = 0
        while not done and step_count < 200:
            for _ in range(min(DRL_CONFIG['actions_per_step'], 50)):
                action = agent.select_action(state, training=False)
                state, _, done, _, info = env_drl.step(action)
                if done:
                    break
            step_count += 1
        
        drl_results['acceptance_ratio'].append(info['acceptance_ratio'])
        drl_results['avg_delay'].append(info['avg_delay'])
        drl_results['resource_util'].append(info['resource_util'])
        
        env_baseline = SFCEnvironment(num_dcs=num_dcs)
        baseline = BaselineHeuristic(env_baseline)
        baseline_info = baseline.run(max_steps=200)
        
        baseline_results['acceptance_ratio'].append(baseline_info['acceptance_ratio'])
        baseline_results['avg_delay'].append(baseline_info['avg_delay'])
        baseline_results['resource_util'].append(baseline_info['resource_util'])
    
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    
    drl_acc = np.mean(drl_results['acceptance_ratio'])
    baseline_acc = np.mean(baseline_results['acceptance_ratio'])
    improvement_acc = ((drl_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    print(f"\nAcceptance Ratio:")
    print(f"  DRL Agent:  {drl_acc:.2%}")
    print(f"  Baseline:   {baseline_acc:.2%}")
    print(f"  Improvement: {improvement_acc:+.2f}%")
    
    drl_delay_vals = [d for d in drl_results['avg_delay'] if d > 0]
    baseline_delay_vals = [d for d in baseline_results['avg_delay'] if d > 0]
    
    if drl_delay_vals and baseline_delay_vals:
        drl_delay = np.mean(drl_delay_vals)
        baseline_delay = np.mean(baseline_delay_vals)
        improvement_delay = ((baseline_delay - drl_delay) / baseline_delay) * 100 if baseline_delay > 0 else 0
        
        print(f"\nAverage E2E Delay:")
        print(f"  DRL Agent:  {drl_delay:.2f} ms")
        print(f"  Baseline:   {baseline_delay:.2f} ms")
        print(f"  Reduction:  {improvement_delay:+.2f}%")
    
    drl_util = np.mean(drl_results['resource_util'])
    baseline_util = np.mean(baseline_results['resource_util'])
    improvement_util = ((baseline_util - drl_util) / baseline_util) * 100 if baseline_util > 0 else 0
    
    print(f"\nResource Utilization:")
    print(f"  DRL Agent:  {drl_util:.2%}")
    print(f"  Baseline:   {baseline_util:.2%}")
    print(f"  Reduction:  {improvement_util:+.2f}%")
    
    print("\n" + "="*60)
    
    return drl_results, baseline_results