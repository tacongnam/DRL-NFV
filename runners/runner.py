import networkx as nx
import config
from core import DataCenter
from envs import SFCEnvironment
from agents import DQNAgent

class Runner:
    graph = None
    dcs = None
    requests = None
    
    @classmethod
    def load_from(cls, file_path):
        from runners.data_loader import load_data

        cls.reset()
        graph, dcs, requests, vnf_specs = load_data(file_path)
        
        config.update_vnf_specs(vnf_specs)
        config.update_resource_constraints(dcs, graph)

        cls.graph = graph
        cls.dcs = dcs
        cls.requests = requests
        
        if requests:
            last_arrival = max(r['arrival_time'] for r in requests)
            max_delay = max(r['max_delay'] for r in requests)
            # Thêm buffer an toàn
            config.MAX_SIM_TIME_PER_EPISODE = last_arrival + max_delay + 0.5
            print(f"  >>> Auto-set Max Sim Time: {config.MAX_SIM_TIME_PER_EPISODE:.1f}")

        print(f"Loaded: {len([d for d in dcs if d.is_server])} servers, {config.NUM_VNF_TYPES} VNFs, {len(requests)} requests")
    
    @classmethod
    def load_from_dict(cls, data_dict):
        cls.reset()
        
        graph = nx.Graph()
        dcs = []
        
        for node_id_str, node_data in data_dict["V"].items():
            node_id = int(node_id_str)
            is_server = node_data.get("server", False)
            
            graph.add_node(node_id, server=is_server, cpu=node_data.get("c_v", 0),
                          ram=node_data.get("r_v", 0), delay=node_data.get("d_v", 0))
            
            if is_server:
                dcs.append(DataCenter(node_id, cpu=node_data.get("c_v"), ram=node_data.get("r_v"),
                                     storage=node_data.get("h_v"), delay=node_data.get("d_v", 0.0),
                                     cost_c=node_data.get("cost_c", 1.0), cost_h=node_data.get("cost_h", 1.0),
                                     cost_r=node_data.get("cost_r", 1.0)))
        
        for link in data_dict["E"]:
            u, v = int(link["u"]), int(link["v"])
            graph.add_edge(u, v, bw=link.get("b_l", 1000), capacity=link.get("b_l", 1000), delay=link.get("d_l", 1.0))
        
        vnf_specs = {idx: {"cpu": vnf.get("c_f", 1.0), "ram": vnf.get("r_f", 1.0), "storage": vnf.get("h_f", 1.0),
                          "startup_time": {int(k): v for k, v in vnf.get("d_f", {}).items()}}
                    for idx, vnf in enumerate(data_dict["F"])}
        
        requests = [{"id": idx, "arrival_time": req.get("T", 0), "source": req["st_r"], "destination": req["d_r"],
                    "vnf_chain": req["F_r"], "bandwidth": req.get("b_r", 1.0), "max_delay": req.get("d_max", 100.0),
                    "type": req.get("type", "Unknown")}
                   for idx, req in enumerate(data_dict["R"])]
        
        dcs.sort(key=lambda x: x.id)
        
        config.update_vnf_specs(vnf_specs)
        config.update_resource_constraints(dcs, graph)

        cls.graph = graph
        cls.dcs = dcs
        cls.requests = requests
    
    @classmethod
    def reset(cls):
        cls.graph = None
        cls.dcs = None
        cls.requests = None
    
    @classmethod
    def create_env(cls, dc_selector=None):
        return SFCEnvironment(cls.graph, cls.dcs, cls.requests, dc_selector)
    
    @classmethod
    def train_dqn_file(cls, file_path, num_updates, dc_selector):
        from runners.train_dqn import train_dqn_on_env
        
        cls.load_from(file_path)
        env = cls.create_env(dc_selector)
        
        if hasattr(env.observation_space, 'spaces'):
            state_shapes = [s.shape for s in env.observation_space.spaces]
        else:
            state_shapes = [env.observation_space.shape]
        
        agent = DQNAgent(state_shapes, env.action_space.n)
        return train_dqn_on_env(cls, env, agent, num_updates, dc_selector)
    
    @classmethod
    def train_dqn_random(cls, num_episodes, dc_selector):
        from runners.train_dqn import train_dqn_random
        return train_dqn_random(cls, num_episodes, dc_selector)
    
    @classmethod
    def collect_and_train_vae(cls, file_path, num_episodes, dc_selector):
        from runners.train_vae import collect_and_train_vae
        cls.load_from(file_path)
        return collect_and_train_vae(cls, num_episodes, dc_selector)
    
    @classmethod
    def train_vae_random(cls, num_episodes, dc_selector, vae_epochs=None):
        from runners.train_vae import collect_and_train_vae_random
        if vae_epochs is None: vae_epochs = config.GENAI_VAE_EPOCHS
        return collect_and_train_vae_random(cls, num_episodes, dc_selector, vae_epochs=vae_epochs)

    @classmethod
    def compare_single_file(cls, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes):
        from runners.compare import compare_single_file
        return compare_single_file(cls, data_file, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes)

    @classmethod
    def compare_all_files(cls, data_folder, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes, filter_str='', smart_sample=False):
        from runners.compare import compare_all_files
        return compare_all_files(cls, data_folder, dqn_model_path, vae_dqn_model_path, vae_model_path, num_episodes, filter_str, smart_sample)