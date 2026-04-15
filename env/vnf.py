import config


class VNF:
    def __init__(self, name: str, h_f: float, c_f: float, r_f: float, d_f: dict = None):
        '''
        :param name:  Id loại VNF
        :param h_f:   Tài nguyên mem
        :param c_f:   Tài nguyên cpu
        :param r_f:   Tài nguyên ram
        :param d_f:   Dict {dc_name: boot_time}. None → wildcard (mọi DC)
        '''
        self.name = name
        self.resource = {"mem": h_f, "cpu": c_f, "ram": r_f}

        # FIX: kiểm tra None hoặc dict rỗng TRƯỚC khi gán, ép key thành str để nhất quán
        if d_f is None or len(d_f) == 0:
            self.d_f = {'-1': 0.0}          # '-1' = wildcard: có thể đặt lên mọi DC
        else:
            self.d_f = {str(k): v for k, v in d_f.items()}

    def get_dcs(self):
        return list(self.d_f.keys())

    def get_requested_resource(self):
        return self.resource


class ListOfVnfs:
    def __init__(self):
        # FIX: dict thay vì list để tra theo key str(idx)
        self.vnfs: dict = {}

    def add_vnf(self, vnf: VNF):
        self.vnfs[str(vnf.name)] = vnf