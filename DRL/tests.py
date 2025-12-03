import config
from env.network import SFCNVEnv
from env.dqn import SFC_DQN
from env.utils import SFCRequest
import numpy as np

def test_config():
    print("1. Testing Config...")
    assert len(config.VNF_TYPES) == 6
    assert config.ACTION_SPACE_SIZE == 2 * 6 + 1
    print("   -> Config OK.")

def test_dqn_shape():
    print("2. Testing DQN Model Shapes...")
    agent = SFC_DQN()
    # Mock inputs đúng kích thước
    s1 = np.zeros((1, 2 * config.NUM_VNF_TYPES + 2))
    s2 = np.zeros((1, config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES)))
    s3 = np.zeros((1, config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES)))
    
    out = agent.model.predict([s1, s2, s3], verbose=0)
    assert out.shape == (1, config.ACTION_SPACE_SIZE)
    print("   -> DQN Shape OK.")

def test_env_logic_success():
    print("3. Testing Environment Logic (Positive Scenario)...")
    env = SFCNVEnv()
    env.reset()
    
    # --- MOCKUP: Tạo giả một Request để test ---
    # Giả sử có 1 request loại 'CloudGaming', cần chuỗi ['NAT', 'FW', ...]
    # Request này đang ở bước 0, tức là cần 'NAT'
    mock_req = SFCRequest(999, 'CloudGaming', 0, 1, 0) 
    env.sfc_manager.active_requests.append(mock_req)
    
    print(f"   Generated Mock Request: Need {mock_req.get_next_vnf()}")

    # --- ACTION: Chọn đúng hành động Phân bổ 'NAT' ---
    # Tìm index của 'NAT' trong danh sách VNF
    nat_idx = config.VNF_TYPES.index('NAT')
    
    # Tính toán action ID cho việc Allocation NAT
    # Action space: [Wait (1)] + [Uninstall (N)] + [Alloc (N)]
    # Index Alloc = 1 + NUM_VNF + vnf_index
    action_alloc_nat = 1 + config.NUM_VNF_TYPES + nat_idx
    
    # Thực hiện bước
    next_s, r, done, _, _ = env.step(action_alloc_nat)
    
    print(f"   Step Result - Reward: {r}")
    
    # --- ASSERTIONS ---
    # 1. Reward phải dương (0.5 cho progress hoặc 2.0 nếu hoàn thành luôn)
    if r > 0:
        print("   -> SUCCESS: Received positive reward for correct allocation.")
    else:
        print(f"   -> FAIL: Expected positive reward, got {r}. Check resource availability.")
        
    # 2. Kiểm tra xem VNF đã được cài vào DC chưa
    dc = env.dcs[env.current_dc_idx - 1] # -1 vì step() đã move sang DC kế tiếp
    installed_types = [v.vnf_type for v in dc.installed_vnfs]
    if 'NAT' in installed_types:
        print("   -> SUCCESS: 'NAT' VNF installed on DC.")
    else:
        print("   -> FAIL: VNF not found on DC.")

if __name__ == "__main__":
    test_config()
    test_dqn_shape()
    test_env_logic_success()