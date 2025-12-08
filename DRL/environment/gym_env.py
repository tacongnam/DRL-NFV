import gymnasium as gym
import numpy as np
import config
from spaces.sfc_manager import SFC_Manager
from spaces.dc import DataCenter
from environment.observer import Observer
from environment.controller import ActionController
from environment.simulator import Simulator
from environment.utils import get_valid_actions_mask

class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        # Định nghĩa Observation Space đúng theo kiến trúc 3 đầu vào
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=np.inf, shape=(2 * config.NUM_VNF_TYPES + 2,), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),), dtype=np.float32)
        ))
        
        # SỬA: Đổi tên thành sfc_manager cho khớp với demo.py
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.simulator = None
        self.controller = None
        self.current_dc_idx = 0 # Init variable
        
    def reset(self, num_dcs=None):
        # 1. Setup DCs
        n = num_dcs if num_dcs else np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(n)]
        
        # 2. Setup Manager & Controller
        self.sfc_manager = SFC_Manager()
        self.sfc_manager.reset_history()
        
        # Khởi tạo Controller & Simulator với tham chiếu mới
        self.controller = ActionController(self.sfc_manager)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        self.simulator.reset()
        
        # 3. Reset loop variables
        # SỬA: Đổi tên thành current_dc_idx cho khớp với demo.py
        self.current_dc_idx = 0
        self.steps_in_ts = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Truyền đúng tham chiếu self.sfc_manager
        return Observer.get_full_state(self.dcs[self.current_dc_idx], self.sfc_manager)

    def _get_valid_actions_mask(self):
        # Hàm helper hỗ trợ training/testing
        return get_valid_actions_mask(
            self.dcs[self.current_dc_idx], 
            self.sfc_manager.active_requests
        )

    def step(self, action):
        curr_dc = self.dcs[self.current_dc_idx]
        
        # 1. Execute Action
        reward, accepted = self.controller.execute_action(action, curr_dc)
        
        # 2. Advance DC Pointer (Round Robin)
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        
        # 3. Time Simulation Logic
        # Advance time after everyone has acted (simplified logic)
        self.steps_in_ts += 1
        if self.steps_in_ts >= config.ACTIONS_PER_TIME_STEP:
            drop_pen = self.simulator.advance_time()
            reward += drop_pen
            self.steps_in_ts = 0
            
        done = self.simulator.is_done()
        
        # 4. Info
        info = {
            'acc_ratio': self.sfc_manager.get_statistics()['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'generated': self.sfc_manager.req_counter
        }
        
        # Return 5 giá trị theo chuẩn Gymnasium mới
        return self._get_obs(), reward, done, False, info