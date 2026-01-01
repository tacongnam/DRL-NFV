import config

class Node:
    def __init__(self, node_id, is_server=False):
        self.id = node_id
        self.is_server = is_server
        self.installed_vnfs = []

    def has_resources(self, vnf_type):
        return False

    def count_vnf_type(self, vnf_type):
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type)

    def count_idle_vnf_type(self, vnf_type):
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type and v.is_idle())


class SwitchNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id, is_server=False)


class DataCenter(Node):
    def __init__(self, dc_id, cpu=None, ram=None, storage=None, delay=None, 
                 cost_c=None, cost_h=None, cost_r=None):
        super().__init__(dc_id, is_server=True)
        
        self.cpu = cpu if cpu is not None else config.DC_CPU_CYCLES
        self.ram = ram if ram is not None else config.DC_RAM
        self.storage = storage if storage is not None else config.DC_STORAGE
        
        self.delay = delay
        self.cost_c = cost_c
        self.cost_h = cost_h
        self.cost_r = cost_r

    def has_resources(self, vnf_type):
        specs = config.VNF_SPECS[vnf_type]
        return (self.cpu >= specs['cpu'] and 
                self.ram >= specs['ram'] and 
                self.storage >= specs['storage'])

    def consume_resources(self, vnf_type):
        if not self.has_resources(vnf_type):
            return False
            
        specs = config.VNF_SPECS[vnf_type]
        self.cpu -= specs['cpu']
        self.ram -= specs['ram']
        self.storage -= specs['storage']
        return True

    def release_resources(self, vnf_type):
        specs = config.VNF_SPECS[vnf_type]
        self.cpu += specs['cpu']
        self.ram += specs['ram']
        self.storage += specs['storage']

    def get_idle_vnf(self, vnf_type):
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                return vnf
        return None

    def uninstall_idle_vnf(self, vnf_type):
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                self.installed_vnfs.remove(vnf)
                self.release_resources(vnf_type)
                return True
        return False