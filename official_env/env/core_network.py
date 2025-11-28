import gymnasium as gym
import numpy as np
from gymnasium import spaces
import config

class CoreNetworkEnv(gym.Env):
    def __init__(self):
        super(CoreNetworkEnv, self).__init__()
        
        # State: Resources [CPU, RAM, Sto] * Num_DCs
        self.dcs_state = np.zeros((config.NUM_DCS, 3)) 
        self.max_resources = np.zeros((config.NUM_DCS, 3))
        
        # Logical Links (Full mesh)
        self.link_bw = np.full((config.NUM_DCS, config.NUM_DCS), config.LINK_BANDWIDTH)
        
        # Action: Place (Num_VNFs), Uninstall (Num_VNFs), Wait (1)
        # [Paper Section IV.B]: Actions include placing, uninstalling, idle wait.
        self.action_space = spaces.Discrete(2 * config.NUM_VNFS + 1)
        
        # Observation: All DC States + Current Request Info
        obs_dim = config.NUM_DCS * 3 + 5 
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        # Init random resources
        for i in range(config.NUM_DCS):
            self.max_resources[i] = [
                np.random.randint(min_v, max_v) 
                for min_v, max_v in zip(config.DC_RESOURCES_MIN, config.DC_RESOURCES_MAX)
            ]
            self.dcs_state[i] = self.max_resources[i]
            
        self.current_sfc = None
        self.current_vnf_idx = 0
        self.placed_vnfs_loc = [] # Tracking: [DC_idx_for_VNF1, DC_idx_for_VNF2...]
        
        return self._get_state()

    def set_current_request(self, sfc_data):
        self.current_sfc = sfc_data
        self.current_vnf_idx = 0
        self.placed_vnfs_loc = []

    def _get_state(self, selected_dc_idx=None):
        """
        Trả về dictionary chứa 3 inputs cho DQN Multi-input
        """
        # 1. Global SFC Info
        if self.current_sfc:
            rem_vnfs = len(self.current_sfc['vnfs']) - self.current_vnf_idx
            current_vnf = self.current_sfc['vnfs'][self.current_vnf_idx] if rem_vnfs > 0 else 0
            global_info = np.array([
                self.current_sfc['bw'] / 100.0,
                self.current_sfc['delay'] / 100.0,
                rem_vnfs,
                current_vnf
            ])
        else:
            global_info = np.zeros(4)
            
        # Nếu chưa chọn DC nào (lúc reset), trả về dummy
        if selected_dc_idx is None:
            return {
                "dc_state_input": np.zeros(3),
                "dc_sfc_input": np.zeros(config.NUM_VNFS),
                "global_sfc_input": global_info
            }

        # 2. Selected DC State
        dc_res = self.dcs_state[selected_dc_idx] / (self.max_resources[selected_dc_idx] + 1e-6)
        
        # 3. DC's SFC State (Giả lập: One-hot vector các loại VNF đã cài trên DC này)
        # Ở đây ta giả lập đơn giản: Nếu DC này đã dùng cho SFC hiện tại thì đánh dấu
        # (Thực tế có thể là danh sách các VNF đã cài sẵn)
        dc_sfc_info = np.zeros(config.NUM_VNFS)
        # Logic tùy chỉnh: ví dụ biểu diễn các VNF types available
        
        return {
            "dc_state_input": dc_res,
            "dc_sfc_input": dc_sfc_info, 
            "global_sfc_input": global_info
        }

    def get_dc_state(self, dc_idx):
        """Trả về state riêng của 1 DC cho VAE encode"""
        norm_res = self.dcs_state[dc_idx] / (self.max_resources[dc_idx] + 1e-6)
        return norm_res

    def step(self, action, selected_dc_idx):
        """
        [Paper Section IV.B]: Environment Update based on Action on Selected DC.
        """
        reward = 0
        done = False
        info = {}
        
        # --- RÀNG BUỘC CỐT LÕI: KHÔNG ĐƯỢC TRÙNG DC TRONG CÙNG SFC ---
        if selected_dc_idx in self.placed_vnfs_loc:
            # Nếu logic bên ngoài (Main) bị lỗi mà vẫn chọn DC cũ -> Phạt nặng
            return self._get_state(), -5.0, True, {"error": "DC Constraint Violation"}

        # Decode Action
        # 0 -> N-1: Place VNF
        # N -> 2N-1: Uninstall
        # 2N: Wait
        
        if action < config.NUM_VNFS: # PLACE VNF
            vnf_type = action
            expected_vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
            
            # Check 1: DQN có chọn đúng loại VNF cần đặt tiếp theo không?
            if vnf_type != expected_vnf:
                reward = -1.0
            else:
                # Check 2: Tài nguyên DC có đủ không?
                req_res = config.VNF_COSTS[vnf_type]
                avail = self.dcs_state[selected_dc_idx]
                
                if all(avail[k] >= req_res[k] for k in range(3)):
                    # Check 3: Băng thông (nếu không phải VNF đầu tiên)
                    bw_ok = True
                    if len(self.placed_vnfs_loc) > 0:
                        prev_dc = self.placed_vnfs_loc[-1]
                        if self.link_bw[prev_dc][selected_dc_idx] < self.current_sfc['bw']:
                            bw_ok = False
                    
                    if bw_ok:
                        # --- SUCCESSFUL PLACEMENT ---
                        # Trừ tài nguyên
                        self.dcs_state[selected_dc_idx] -= req_res
                        if len(self.placed_vnfs_loc) > 0:
                            prev_dc = self.placed_vnfs_loc[-1]
                            self.link_bw[prev_dc][selected_dc_idx] -= self.current_sfc['bw']
                            self.link_bw[selected_dc_idx][prev_dc] -= self.current_sfc['bw']
                        
                        # Cập nhật tracking
                        self.placed_vnfs_loc.append(selected_dc_idx)
                        self.current_vnf_idx += 1
                        
                        reward = 1.0 # Reward nhỏ cho tiến bộ
                        
                        # Kiểm tra đã xong SFC chưa
                        if self.current_vnf_idx >= len(self.current_sfc['vnfs']):
                            reward = 5.0 # [Paper]: Positive reward for fulfilling SFC
                            done = True # Request completed
                    else:
                        reward = -1.5 # Lỗi BW
                        done = True # Drop Request
                else:
                    reward = -1.5 # Lỗi Resource
                    done = True # Drop Request

        elif action == 2 * config.NUM_VNFS: # WAIT
            reward = -0.1
        else: # UNINSTALL
            reward = -0.5

        next_state_dict = self._get_state(selected_dc_idx)
        return next_state_dict, reward, done, info