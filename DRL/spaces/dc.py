import config

class DataCenter:
    def __init__(self, dc_id):
        self.id = dc_id
        self.cpu = config.DC_CPU_CYCLES
        self.ram = config.DC_RAM
        self.storage = config.DC_STORAGE
        self.installed_vnfs = [] # List of VNFInstance objects

    def has_resources(self, vnf_type):
        spec = config.VNF_SPECS[vnf_type]
        return (self.cpu >= spec['cpu'] and 
                self.ram >= spec['ram'] and 
                self.storage >= spec['storage'])

    def consume_resources(self, vnf_type):
        spec = config.VNF_SPECS[vnf_type]
        self.cpu -= spec['cpu']
        self.ram -= spec['ram']
        self.storage -= spec['storage']

    def release_resources(self, vnf_type):
        spec = config.VNF_SPECS[vnf_type]
        self.cpu += spec['cpu']
        self.ram += spec['ram']
        self.storage += spec['storage']