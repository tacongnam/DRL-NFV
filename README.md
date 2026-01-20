# DRL-based Network Function Virtualization Placement

Hệ thống sử dụng Deep Reinforcement Learning (DQN + VAE) để giải quyết bài toán đặt Virtual Network Functions (VNFs) trong môi trường NFV/SDN.

## 📁 Cấu trúc thư mục
```
DRL-NFV/
├── agents/                          # Các agent DRL
│   ├── dqn_agent.py                # Deep Q-Network agent
│   ├── vae_agent.py                # Variational Autoencoder agent
│   └── vae_trainer.py              # Trainer cho VAE
│
├── core/                            # Core logic của simulator
│   ├── dc.py                       # DataCenter và SwitchNode
│   ├── request.py                  # SFC Request
│   ├── sfc_manager.py              # Quản lý requests
│   ├── simulator.py                # Discrete-event simulator
│   ├── topology.py                 # Network topology manager
│   └── vnf.py                      # VNF Instance
│
├── envs/                            # Gym Environment
│   ├── action_handler.py           # Xử lý actions (allocate/uninstall)
│   ├── debug_tracker.py            # Debug và tracking
│   ├── env.py                      # SFCEnvironment (Gym interface)
│   ├── observer.py                 # State observation
│   ├── request_selector.py         # Chọn request ưu tiên
│   ├── selectors.py                # DC ordering strategies
│   └── utils.py                    # Action mask và utilities
│
├── runners/                         # Training và testing pipelines
│   ├── compare.py                  # So sánh DQN vs VAE-DQN
│   ├── data_generator.py           # Generate random scenarios
│   ├── data_loader.py              # Load từ JSON files
│   ├── runner.py                   # Core Runner class
│   ├── train_dqn.py                # Training DQN
│   └── train_vae.py                # Collect & train VAE
│
├── data/                            # Test datasets (30+ files)
│   ├── cogent_centers_atlanta_easy_s1.json
│   ├── cogent_centers_atlanta_medium_s1.json
│   ├── cogent_centers_atlanta_hard_s1.json
│   └── ...                         # More locations & difficulties
│
├── models/                          # Saved models
│   ├── best_model_q.weights.h5    # DQN model
│   ├── vae_model_*.weights.h5      # VAE models
│   └── checkpoint_*.weights.h5     # Checkpoints
│
├── fig/                             # Output figures
│   ├── comparison_grouped.png      # Grouped by location & difficulty
│   ├── comparison_by_difficulty.png # By difficulty level
│   └── comparison_by_location.png  # By location
│
├── config.py                        # Global configuration
├── main.py                          # CLI entry point
└── README.md
```

## 🎯 Ý tưởng thuật toán

### Bài toán
Đặt các Virtual Network Functions (VNFs) lên các Data Centers để phục vụ Service Function Chain (SFC) requests với mục tiêu:
- **Maximize**: Acceptance Ratio (số requests được phục vụ)
- **Minimize**: End-to-End Delay, Resource consumption
- **Constraints**: CPU, RAM, Storage, Bandwidth, Latency

### Kiến trúc DRL

#### 1. Deep Q-Network (DQN)
- **State**: 3 inputs
  - DC State: `[CPU, RAM, installed_VNFs, idle_VNFs]`
  - DC-Demand State: `[VNF_demand, chain_patterns]`
  - Global State: `[total_requests, avg_delay, global_VNF_demand]`
  
- **Action Space**: `2V + 1` (V = MAX_VNF_TYPES = 10)
  - Action 0: WAIT
  - Actions 1→10: UNINSTALL VNF type 0-9
  - Actions 11→20: ALLOCATE VNF type 0-9

- **Reward**:
  - `+2.0`: SFC completed
  - `-1.5`: SFC dropped (timeout)
  - `-1.0`: Invalid action
  - `-0.5`: Uninstall needed VNF
  - `0.0`: Otherwise

- **Network Architecture**:
```
  Input1 [DC] → Dense(32) →
  Input2 [Demand] → Dense(64) → Concat → Attention → Dense(96) → Dense(64) → Q-values
  Input3 [Global] → Dense(64) →
```

#### 2. VAE-enhanced DQN
- **VAE Encoder**: DC_State → Latent representation (32D)
- **VAE Decoder**: Latent → Next_DC_State (prediction)
- **Value Network**: Latent → DC priority score
- **Benefit**: DCs được sắp xếp theo value từ VAE thay vì heuristic priority

### Reconfigurability
- **Padding scheme**: State size cố định với `MAX_VNF_TYPES=10`
- **Flexible training**: VNF types từ 2-10 trong mỗi episode
- **No retraining needed**: Model hoạt động với bất kỳ số VNF types nào (2-10)

## 🚀 Pipeline đầy đủ

### Pipeline tự động (Khuyến nghị)
```bash
python main.py train pipeline --episodes 500 --vae-episodes 200
```

**Chức năng:**
1. **Train DQN** với random scenarios (500 episodes)
2. **Collect VAE data** từ DQN đã train (200 episodes)
3. **Train VAE models** (Encoder, Decoder, Value Network)

**Output:**
```
