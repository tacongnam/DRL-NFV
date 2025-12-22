# spaces/dc.py
from DRL import config

from DRL.core.vnf import VNFInstance

class DataCenter:
    """Đại diện cho một Data Center"""
    
    def __init__(self, dc_id, cpu=None, ram=None, storage=None, delay=None, cost_c=None, cost_h=None, cost_r=None):
        self.id = dc_id
        self.cpu = cpu if cpu is not None else config.DC_CPU_CYCLES
        self.ram = ram if ram is not None else config.DC_RAM
        self.storage = storage if storage is not None else config.DC_STORAGE
        self.delay = delay
        self.cost_c = cost_c
        self.cost_h = cost_h
        self.cost_r = cost_r
        self.installed_vnfs = []  # List of VNFInstance
        
    def has_resources(self, vnf_type):
        """Kiểm tra DC có đủ tài nguyên để cài đặt VNF mới"""
        specs = config.VNF_SPECS[vnf_type]
        return (self.cpu >= specs['cpu'] and 
                self.ram >= specs['ram'] and 
                self.storage >= specs['storage'])

    def consume_resources(self, vnf_type):
        """
        Tiêu thụ tài nguyên để cài đặt VNF mới
        Trả về True nếu thành công
        """
        if self.has_resources(vnf_type):
            specs = config.VNF_SPECS[vnf_type]
            self.cpu -= specs['cpu']
            self.ram -= specs['ram']
            self.storage -= specs['storage']
            return True
        return False

    def release_resources(self, vnf_type):
        """Giải phóng tài nguyên khi uninstall VNF"""
        specs = config.VNF_SPECS[vnf_type]
        self.cpu += specs['cpu']
        self.ram += specs['ram']
        self.storage += specs['storage']

    def get_idle_vnf(self, vnf_type):
        """
        Tìm một VNF instance đang rảnh của loại vnf_type
        Cơ chế tái sử dụng: Ưu tiên VNF đã cài đặt sẵn
        """
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                return vnf
        return None

    def count_vnf_type(self, vnf_type):
        """Đếm số lượng VNF instance của một loại"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type)

    def count_idle_vnf_type(self, vnf_type):
        """Đếm số lượng VNF instance đang rảnh của một loại"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type and v.is_idle())

    def uninstall_idle_vnf(self, vnf_type):
        """
        Gỡ bỏ một VNF instance đang rảnh
        Trả về True nếu thành công
        """
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                self.installed_vnfs.remove(vnf)
                self.release_resources(vnf_type)
                return True
        return False