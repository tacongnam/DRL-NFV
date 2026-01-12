import random
import numpy as np

class DataGenerator:
    @staticmethod
    def generate_scenario(num_dcs_range=(2, 10), num_switches_range=(2, 6), 
                         num_vnf_types=2, num_requests_range=(5, 20)):
        num_dcs = random.randint(*num_dcs_range)
        num_switches = random.randint(*num_switches_range)
        total_nodes = num_dcs + num_switches
        
        # Enhanced DC resources for better capacity
        nodes = {}
        for i in range(num_dcs):
            nodes[str(i)] = {
                "server": True,
                "c_v": random.randint(80, 120),  # Increased from 20-100
                "r_v": random.randint(150, 256), # Increased from 30-200
                "h_v": random.randint(1500, 2048), # Increased from 20-150
                "d_v": round(random.uniform(0.01, 0.3), 2), # Reduced from 0.5
                "cost_c": random.randint(80, 150),
                "cost_r": random.randint(50, 100),
                "cost_h": random.randint(15, 30)
            }
        
        for i in range(num_dcs, total_nodes):
            nodes[str(i)] = {"server": False}
        
        # Enhanced network connectivity
        edges = []
        all_nodes = list(range(total_nodes))
        
        # Each DC connects to 3-5 nodes (better connectivity)
        for i in range(num_dcs):
            num_connections = min(random.randint(3, 5), total_nodes - 1)
            targets = random.sample([n for n in all_nodes if n != i], num_connections)
            for j in targets:
                if i < j:
                    edges.append({
                        "u": i, "v": j,
                        "b_l": random.randint(100, 250), # Increased bandwidth
                        "d_l": round(random.uniform(0.01, 0.08), 2)
                    })
        
        # Switches connect to 2-4 nodes
        for i in range(num_dcs, total_nodes):
            num_connections = min(random.randint(2, 4), total_nodes - 1)
            targets = random.sample([n for n in all_nodes if n != i], num_connections)
            for j in targets:
                if i < j:
                    edges.append({
                        "u": i, "v": j,
                        "b_l": random.randint(80, 200),
                        "d_l": round(random.uniform(0.01, 0.06), 2)
                    })
        
        # VNF specs with realistic resource requirements
        vnf_specs = []
        for _ in range(num_vnf_types):
            vnf_specs.append({
                "c_f": round(random.uniform(1.0, 2.0), 1),
                "r_f": round(random.uniform(1.0, 2.0), 1),
                "h_f": round(random.uniform(0.8, 1.5), 1),
                "d_f": {str(i): round(random.uniform(0.03, 0.08), 2) 
                       for i in range(random.randint(2, 4))}
            })
        
        # CRITICAL: Enhanced request generation with better distribution
        num_requests = random.randint(*num_requests_range)
        requests = DataGenerator._generate_distributed_requests(
            num_requests, num_vnf_types, total_nodes
        )
        
        return {"V": nodes, "E": edges, "F": vnf_specs, "R": requests}
    
    @staticmethod
    def _generate_distributed_requests(num_requests, num_vnf_types, total_nodes):
        """Generate requests with Poisson arrival and realistic deadlines"""
        requests = []
        
        # Use Poisson process for arrival times (more realistic)
        max_time = 1000  # Simulation time
        lambda_rate = num_requests / max_time  # Average arrival rate
        
        arrival_times = []
        current_time = 0.0
        
        # Generate Poisson arrival times
        for _ in range(num_requests):
            # Exponential inter-arrival time
            inter_arrival = np.random.exponential(1.0 / lambda_rate)
            current_time += inter_arrival
            if current_time < max_time:
                arrival_times.append(int(current_time))
            else:
                break
        
        # If not enough requests, fill remaining with uniform distribution
        while len(arrival_times) < num_requests:
            arrival_times.append(random.randint(0, max_time - 1))
        
        arrival_times.sort()
        
        for i, arrival_t in enumerate(arrival_times[:num_requests]):
            chain_length = random.randint(1, min(4, num_vnf_types))
            chain = [random.randint(0, num_vnf_types - 1) for _ in range(chain_length)]
            
            # Bandwidth: realistic distribution (most are low, some are high)
            if random.random() < 0.7:  # 70% low bandwidth
                bandwidth = round(random.uniform(0.5, 2.0), 1)
            else:  # 30% high bandwidth
                bandwidth = round(random.uniform(2.0, 5.0), 1)
            
            # Calculate realistic max_delay based on chain complexity
            base_delay = chain_length * 15.0  # Base: 15ms per VNF
            
            # Add propagation delay estimate (assume 2-3 hops average)
            prop_delay_estimate = random.uniform(0.5, 2.0)
            
            # Add processing delay estimate
            proc_delay_estimate = chain_length * random.uniform(0.5, 1.5)
            
            # Total minimum required delay
            min_required = base_delay + prop_delay_estimate + proc_delay_estimate
            
            # Add slack time (50% to 200% of minimum)
            slack_multiplier = random.uniform(1.5, 3.0)
            max_delay = round(min_required * slack_multiplier, 1)
            
            # Ensure reasonable bounds
            max_delay = max(20.0, min(max_delay, 200.0))
            
            # Source and destination
            source = random.randint(0, total_nodes - 1)
            dest = random.randint(0, total_nodes - 1)
            while dest == source:
                dest = random.randint(0, total_nodes - 1)
            
            requests.append({
                "T": arrival_t,
                "st_r": source,
                "d_r": dest,
                "F_r": chain,
                "b_r": bandwidth,
                "d_max": max_delay
            })
        
        return requests