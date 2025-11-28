import unittest
import config
from env.core_network import CoreNetworkEnv

class TestSystem(unittest.TestCase):
    def test_different_dc_constraint(self):
        """Kiểm tra logic cấm đặt VNF trùng DC trong cùng SFC"""
        env = CoreNetworkEnv()
        
        # Tạo request Ind4.0 (2 VNFs: index 0, 1)
        sfc_data = config.SFC_TYPES['Ind4.0'].copy()
        env.set_current_request(sfc_data)
        
        # 1. Đặt VNF đầu tiên vào DC 0 -> Thành công
        _, r1, _, _ = env.step(action=0, selected_dc_idx=0)
        self.assertGreater(r1, 0)
        self.assertIn(0, env.placed_vnfs_loc)
        
        # 2. Cố tình đặt VNF thứ hai cũng vào DC 0 -> Phải thất bại nặng
        _, r2, done, info = env.step(action=1, selected_dc_idx=0)
        
        # Reward phải rất thấp (-5.0 trong code) và Done = True
        self.assertEqual(r2, -5.0) 
        self.assertTrue(done)
        self.assertIn("Violation", info.get('error', ''))

if __name__ == '__main__':
    unittest.main()