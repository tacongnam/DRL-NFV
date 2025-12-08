import config

class VNFInstance:
    def __init__(self, vnf_type, dc_id):
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.remaining_proc_time = 0
        self.assigned_sfc_id = None

    def is_idle(self):
        return self.remaining_proc_time <= 0

    def assign(self, sfc_id, proc_time):
        self.assigned_sfc_id = sfc_id
        self.remaining_proc_time = proc_time

    def tick(self):
        if self.remaining_proc_time > 0:
            self.remaining_proc_time -= config.TIME_STEP
            if self.remaining_proc_time < 0: 
                self.remaining_proc_time = 0
            if self.remaining_proc_time == 0:
                self.assigned_sfc_id = None