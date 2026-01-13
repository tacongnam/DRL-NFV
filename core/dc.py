import config
from core.vnf import VNFInstance

class Node:
    """Lớp cha cho các node trong mạng."""
    def __init__(self, node_id, is_server=False):
        self.id = node_id
        self.is_server = is_server

class SwitchNode(Node):
    """Switch chỉ làm nhiệm vụ trung chuyển, không chạy VNF."""
    def __init__(self, node_id):
        super().__init__(node_id, is_server=False)

class DataCenter(Node):
    """Server/DC có khả năng tính toán và lưu trữ VNF."""
    
    def __init__(self, dc_id, cpu=None, ram=None, storage=None, delay=None, 
                 cost_c=None, cost_h=None, cost_r=None):
        super().__init__(dc_id, is_server=True)
        
        self.cpu = cpu if cpu is not None else config.MAX_CPU
        self.ram = ram if ram is not None else config.MAX_RAM
        self.storage = storage if storage is not None else config.MAX_STORAGE
        
        # Tổng tài nguyên ban đầu (để tính toán % sử dụng nếu cần)
        self.initial_resources = {'cpu': self.cpu, 'ram': self.ram, 'storage': self.storage}
        
        self.delay = delay  # Processing delay nội tại của DC
        self.cost_c = cost_c
        self.cost_h = cost_h
        self.cost_r = cost_r
        
        self.installed_vnfs = [] # List[VNFInstance]

    def get_state(self):
        """Trả về trạng thái hiện tại của DC (Normalized nếu cần)."""
        return [self.cpu, self.ram, self.storage, len(self.installed_vnfs)]

    def _get_idle_vnf(self, vnf_type):
        for v in self.installed_vnfs:
            if v.vnf_type == vnf_type and v.is_idle():
                return v
        return None
    
    def has_resources(self, vnf_type):
        """Kiểm tra xem có đủ tài nguyên cài đặt VNF mới không."""
        specs = config.VNF_SPECS[vnf_type]
        return (self.cpu >= specs['cpu'] and 
                self.ram >= specs['ram'] and 
                self.storage >= specs['storage'])

    def _consume_resources(self, vnf_type):
        specs = config.VNF_SPECS[vnf_type]
        self.cpu -= specs['cpu']
        self.ram -= specs['ram']
        self.storage -= specs['storage']

    def _release_resources(self, vnf_type):
        specs = config.VNF_SPECS[vnf_type]
        self.cpu += specs['cpu']
        self.ram += specs['ram']
        self.storage += specs['storage']

    def install_vnf(self, vnf_type):
        """Cài đặt VNF mới."""
        if self.has_resources(vnf_type):
            new_vnf = VNFInstance(vnf_type, self.id)
            self.installed_vnfs.append(new_vnf)
            self._consume_resources(vnf_type)
            return new_vnf
        return None

    def uninstall_vnf(self, vnf_type):
        """Gỡ cài đặt VNF rảnh."""
        vnf = self._get_idle_vnf(vnf_type)
        if vnf:
            self._release_resources(vnf_type)
            self.installed_vnfs.remove(vnf)
            return True
        return False