import gymnasium as gym
import numpy as np
import config
from core.sfc_manager import SFC_Manager
from core.dc import DataCenter
from core.topology import TopologyManager
from envs.observer import Observer
from envs.controller import ActionController
from core.simulator import Simulator
from envs.utils import get_valid_actions_mask

class SFCBaseEnv(gym.Env):
    """
    Base Environment cho SFC Provisioning.
    Chứa logic chung cho cả DRL thuần và VAE-assisted DRL.
    """
    
    def __init__(self):
        super().__init__()
        
        # --- 1. Define Spaces (Chung) ---
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(2 * config.NUM_VNF_TYPES + 2,), 
                          dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),), 
                          dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, 
                          shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),), 
                          dtype=np.float32)
        ))
        
        # --- 2. Core Components (Placeholder) ---
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.topology = None
        self.simulator = None
        self.controller = None
        
        # --- 3. State Variables ---
        self.count_step = 0
        self.current_dc_idx = 0
        self.dc_order = []  # List chứa thứ tự DC sẽ duyệt qua (VAE hoặc Priority quyết định)
        self.actions_this_step = 0

    def reset(self, num_dcs=None, seed=None):
        """Reset environment logic chung"""
        super().reset(seed=seed)
        
        # Setup DCs
        n = num_dcs if num_dcs else np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(n)]
        
        # Setup topology
        self.topology = TopologyManager(n)
        
        # Setup managers
        self.sfc_manager = SFC_Manager()
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        
        # Hook cho class con khởi tạo thêm (vd: PriorityManager)
        self._after_init_components() 
        
        self.simulator.reset()
        
        # Generate initial traffic
        if self.simulator.should_generate_initial_traffic():
            self.sfc_manager.generate_requests(0, len(self.dcs))
            self.simulator.has_generated_initial = True
        
        # Reset state logic
        self._update_dc_order() # Class con sẽ định nghĩa hàm này
        self.current_dc_idx = 0
        self.actions_this_step = 0
        self.count_step = 0
        
        # Hook reset state riêng (vd: clear prev_states của VAE)
        self._reset_specific_state()
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Logic step chung.
        Sử dụng Template Method Pattern với các hooks.
        """
        # Hook: Trước khi thực hiện action (VAE dùng để snapshot state)
        self._pre_action_hook()
        
        # Get current DC
        if not self.dc_order:
             self._update_dc_order()
             
        curr_dc_id = self.dc_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        # Execute action
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        # Move to next DC
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        self.actions_this_step += 1
        
        # Advance time logic
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            # Cập nhật lại thứ tự DC (Random/VAE hoặc Priority)
            self._update_dc_order()
            self.current_dc_idx = 0
        
        # Check done
        done = self.simulator.is_done()
        self.count_step += 1
        
        # Info
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay']
        }
        
        # Hook: Thêm info riêng (VAE thêm prev_states)
        self._post_step_info_hook(info)
        
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """Lấy observation dựa trên dc_order hiện tại"""
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return Observer.get_dc_state(curr_dc, self.sfc_manager, None)

    def _get_valid_actions_mask(self):
        """Lấy mask dựa trên dc_order hiện tại"""
        if not self.dc_order:
            self._update_dc_order()
        curr_dc = self.dcs[self.dc_order[self.current_dc_idx]]
        return get_valid_actions_mask(curr_dc, self.sfc_manager.active_requests)

    # --- Abstract Methods / Hooks (Class con cần override) ---
    def _update_dc_order(self):
        """Xác định thứ tự duyệt DC (VAE vs Priority)"""
        raise NotImplementedError

    def _after_init_components(self):
        """Khởi tạo thêm manager nếu cần"""
        pass

    def _reset_specific_state(self):
        """Reset biến state riêng"""
        pass

    def _pre_action_hook(self):
        """Chạy trước khi execute action"""
        pass
        
    def _post_step_info_hook(self, info):
        """Thêm dữ liệu vào info"""
        pass