import config

class DataCenter:
    def __init__(self, dc_id):
        self.id = dc_id
        self.cpu = config.DC_CPU_CYCLES
        self.ram = config.DC_RAM
        self.storage = config.DC_STORAGE
        self.installed_vnfs = []
        
    def has_resources(self, vnf_type):
        s = config.VNF_SPECS[vnf_type]
        return (self.cpu >= s['cpu'] and self.ram >= s['ram'] and self.storage >= s['storage'])

    def consume_resources(self, vnf_type):
        if self.has_resources(vnf_type):
            s = config.VNF_SPECS[vnf_type]
            self.cpu -= s['cpu']
            self.ram -= s['ram']
            self.storage -= s['storage']
            return True
        return False

    def release_resources(self, vnf_type):
        s = config.VNF_SPECS[vnf_type]
        self.cpu += s['cpu']
        self.ram += s['ram']
        self.storage += s['storage']