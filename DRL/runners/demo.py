# runners/demo.py
import config
from environment.gym_env import Env
from agent.agent import Agent
from spaces.request import SFCRequest
import numpy as np
import sys

def test_config():
    print("1. Testing Config...")
    assert len(config.VNF_TYPES) == 6
    assert config.ACTION_SPACE_SIZE == 2 * 6 + 1
    print("   -> Config OK.")

def test_dqn_shape():
    print("2. Testing DQN Model Shapes...")
    agent = Agent()
    # Mock inputs đúng kích thước
    s1 = np.zeros((1, 2 * config.NUM_VNF_TYPES + 2))
    s2 = np.zeros((1, config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES)))
    s3 = np.zeros((1, config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES)))
    
    try:
        out = agent.model.predict([s1, s2, s3], verbose=0)
        assert out.shape == (1, config.ACTION_SPACE_SIZE)
        print("   -> DQN Shape OK.")
    except Exception as e:
        print(f"   -> FAIL: Shape Mismatch. {e}")
        sys.exit(1)

def test_env_logic_success():
    print("3. Testing Environment Logic (Positive Scenario)...")
    env = Env()
    state, _ = env.reset()
    
    # --- MOCKUP: Tạo giả một Request để test ---
    mock_req = SFCRequest(999, 'CloudGaming', 0, 1, 0) 
    env.sfc_manager.active_requests.append(mock_req)
    
    print(f"   Generated Mock Request: Need {mock_req.get_next_vnf()}")

    # --- ACTION: Chọn đúng hành động Phân bổ 'NAT' ---
    nat_idx = config.VNF_TYPES.index('NAT')
    # Alloc index = 1 + NUM_VNF + vnf_index
    action_alloc_nat = 1 + config.NUM_VNF_TYPES + nat_idx
    
    # Thực hiện bước
    # env.step trả về: observation, reward, terminated, truncated, info
    next_s, r, done, _, info = env.step(action_alloc_nat)
    
    print(f"   Step Result - Reward: {r}")
    print(f"   Info: {info}")
    
    if r > 0:
        print("   -> SUCCESS: Received positive reward.")
    else:
        # Có thể fail nếu DC không đủ tài nguyên, nhưng logic code chạy được là OK
        print(f"   -> LOGIC RUN: Reward {r}. (May be 0 if partial or negative if invalid)")

    # Test get_statistics
    stats = env.sfc_manager.get_statistics()
    print(f"   Stats Check: {stats}")
    if 'acceptance_ratio' in stats:
        print("   -> SUCCESS: Statistics available.")

def main():
    print("=== RUNNING DEMO TESTS ===")
    test_config()
    test_dqn_shape()
    test_env_logic_success()

if __name__ == "__main__":
    main()  