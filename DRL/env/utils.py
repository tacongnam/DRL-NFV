import numpy as np
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

class SFCRequest:
    def __init__(self, req_id, sfc_type, source, destination, arrival_time):
        self.id = req_id
        self.type = sfc_type
        self.specs = config.SFC_SPECS[sfc_type]
        self.chain = self.specs['chain'][:]
        self.current_vnf_index = 0
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        
        self.max_delay = self.specs['delay'] # D^s
        self.elapsed_time = 0                # TE^s
        self.is_dropped = False
        self.is_completed = False
        
        # Lưu lịch sử đặt: List of tuple (vnf_name, dc_id)
        self.placed_vnfs = [] 

    def get_next_vnf(self):
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None

    def advance_chain(self, dc_id):
        vnf = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf, dc_id))
        self.current_vnf_index += 1
        if self.current_vnf_index >= len(self.chain):
            self.is_completed = True

    def update_time(self):
        self.elapsed_time += config.TIME_STEP
        # Bài báo: Dropped if propagation + processing > D^s. 
        # Ở đây ta check tổng thời gian đã trôi qua.
        if self.elapsed_time > self.max_delay:
            self.is_dropped = True

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

def calculate_priority(sfc_req, dc_id):
    """
    Tính toán Priority P = P1 + P2 + P3 theo Phần III.B của nghiên cứu.
    """
    
    # --- P1: Time Elapsed vs Delay Constraint ---
    # Formula: P1 = TE^s - D^s
    # Ý nghĩa: Giá trị âm, tăng dần về 0 khi thời gian trôi qua.
    p1 = sfc_req.elapsed_time - sfc_req.max_delay

    # --- P2: SFC Chain Location Affinity ---
    # Bài báo: "If any of the VNFs... installed in current DC... high priority"
    # "degrade that VNF's priority" if allocated to other DCs.
    p2 = 0.0
    if len(sfc_req.placed_vnfs) > 0:
        # Lấy VNF vừa được đặt trước đó (Predecessor)
        _, last_dc_id = sfc_req.placed_vnfs[-1]
        
        if last_dc_id == dc_id:
            # Nếu VNF trước đó nằm ở DC này -> Ưu tiên cao để giảm delay truyền dẫn
            p2 = config.PRIORITY_P2_SAME_DC
        else:
            # Nếu VNF trước đó nằm ở DC khác -> Giảm ưu tiên
            p2 = config.PRIORITY_P2_DIFF_DC
    else:
        # Nếu đây là VNF đầu tiên của chuỗi
        # Kiểm tra xem DC này có gần Source của Request không (Optional logic)
        # Trong bài báo chỉ nói về "previous VNFs", nên ta để 0.
        p2 = 0.0

    # --- P3: Urgency ---
    # Formula: P3 = C / (D^s - TE^s + epsilon)
    # Chỉ tính khi thời gian còn lại nhỏ hơn ngưỡng Thr
    remaining_time = sfc_req.max_delay - sfc_req.elapsed_time
    p3 = 0.0
    
    # Nếu thời gian còn lại ít hơn ngưỡng quy định
    if remaining_time < config.URGENCY_THRESHOLD:
        # Tránh chia cho 0 bằng EPSILON (hoặc số âm nếu đã quá hạn)
        denominator = max(remaining_time, config.EPSILON) 
        p3 = config.URGENCY_CONSTANT_C / denominator
    
    # Tổng hợp
    priority = p1 + p2 + p3
    return priority