# spaces/dc.py
from DRL import config
from DRL.core.vnf import VNFInstance

class DataCenter:
    """Represents a Data Center or network node"""
    
    def __init__(self, dc_id, cpu=None, ram=None, storage=None, delay=None, 
                 cost_c=None, cost_h=None, cost_r=None, is_server=True):
        """
        Initialize a Data Center.
        
        Args:
            dc_id: Data center ID
            cpu: CPU capacity (None for non-server nodes)
            ram: RAM capacity (None for non-server nodes)
            storage: Storage capacity (None for non-server nodes)
            delay: Processing delay
            cost_c: CPU cost
            cost_h: Storage cost
            cost_r: RAM cost
            is_server: Whether this node can host VNFs
        """
        self.id = dc_id
        self.is_server = is_server
        
        if is_server:
            self.cpu = cpu if cpu is not None else config.DC_CPU_CYCLES
            self.ram = ram if ram is not None else config.DC_RAM
            self.storage = storage if storage is not None else config.DC_STORAGE
            self.delay = delay
            self.cost_c = cost_c
            self.cost_h = cost_h
            self.cost_r = cost_r
            self.installed_vnfs = []  # List of VNFInstance
        else:
            # Non-server node: cannot deploy VNFs
            self.cpu = None
            self.ram = None
            self.storage = None
            self.delay = None
            self.cost_c = None
            self.cost_h = None
            self.cost_r = None
            self.installed_vnfs = []
        
    def has_resources(self, vnf_type):
        """Check if DC has enough resources to install new VNF"""
        if not self.is_server:
            return False
            
        specs = config.VNF_SPECS[vnf_type]
        return (self.cpu >= specs['cpu'] and 
                self.ram >= specs['ram'] and 
                self.storage >= specs['storage'])

    def consume_resources(self, vnf_type):
        """
        Consume resources to install new VNF.
        Returns True if successful.
        """
        if not self.is_server or not self.has_resources(vnf_type):
            return False
            
        specs = config.VNF_SPECS[vnf_type]
        self.cpu -= specs['cpu']
        self.ram -= specs['ram']
        self.storage -= specs['storage']
        return True

    def release_resources(self, vnf_type):
        """Release resources when uninstalling VNF"""
        if not self.is_server:
            return
            
        specs = config.VNF_SPECS[vnf_type]
        self.cpu += specs['cpu']
        self.ram += specs['ram']
        self.storage += specs['storage']

    def get_idle_vnf(self, vnf_type):
        """
        Find an idle VNF instance of specified type.
        Reuse mechanism: Prioritize already installed VNFs.
        """
        if not self.is_server:
            return None
            
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                return vnf
        return None

    def count_vnf_type(self, vnf_type):
        """Count number of VNF instances of a type"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type)

    def count_idle_vnf_type(self, vnf_type):
        """Count number of idle VNF instances of a type"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type and v.is_idle())

    def uninstall_idle_vnf(self, vnf_type):
        """
        Remove an idle VNF instance.
        Returns True if successful.
        """
        if not self.is_server:
            return False
            
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                self.installed_vnfs.remove(vnf)
                self.release_resources(vnf_type)
                return True
        return False