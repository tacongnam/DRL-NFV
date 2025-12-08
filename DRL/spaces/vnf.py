import config

class VNFInstance:
    def __init__(self, vnf_type, dc_id):
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.specs = config.VNF_SPECS[vnf_type]
        self.remaining_proc_time = 0
        self.assigned_sfc_id = None # ID của SFC đang dùng VNF này

    def is_idle(self):
        return self.remaining_proc_time <= 0

    def assign(self, sfc_id, proc_time):
        self.assigned_sfc_id = sfc_id
        self.remaining_proc_time = proc_time

    def tick(self):
        if self.remaining_proc_time > 0:
            self.remaining_proc_time -= 1