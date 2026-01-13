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
│   ├── statistics.py               # Thống kê metrics
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
================================================================================
FULL TRAINING PIPELINE
================================================================================
Step 1: Train DQN
Step 2: Collect VAE data
Step 3: Train VAE models
================================================================================

>>> STEP 1: Training DQN with random scenarios...
Episode 1/500 [DC:4 SW:14 VNF:6 Req:25]: R=450 AR=76.0%
...
Checkpoint 50: AR=85.34%
...

>>> STEP 2: Collecting VAE data using trained DQN...
Episode 10/200: 45230 samples
...

>>> Training VAE (234567 samples)...
    Epoch 5/50 - Loss: 0.4523
    ...

>>> Training Value Network (234567 samples)...
    Epoch 5/100 - Loss: 0.2134
    ...

================================================================================
PIPELINE COMPLETE!
  DQN model: models/best_model
  VAE model: models/vae_model
================================================================================
```

---

### Pipeline thủ công (Optional)

#### Step 1: Train DQN
```bash
python main.py train dqn --episodes 400
```

#### Step 2: Train VAE
```bash
python main.py train vae --vae-episodes 150
```

---

### So sánh DQN vs VAE-DQN

#### Trên tất cả files (Mặc định)
```bash
python main.py compare
```

**Chức năng:**
- Test trên **TẤT CẢ** files trong `data/` (30+ files)
- Tính toán: Acceptance Ratio, E2E Delay, Throughput
- Phân tích theo:
  - **Location**
  - **Difficulty** (Easy, Medium, Hard)
  - **Combined** (Location + Difficulty)

**Output:**
```
================================================================================
Comparing DQN vs VAE-DQN on all files in data/
================================================================================

Testing 30 files...

File 1/30: cogent_centers_atlanta_easy_s1.json
  DQN:     AR=92.3% Delay=47.5ms TP=245.6
  VAE-DQN: AR=95.1% Delay=43.2ms TP=267.3

File 2/30: cogent_centers_atlanta_medium_s1.json
  DQN:     AR=89.7% Delay=51.3ms TP=223.4
  VAE-DQN: AR=93.4% Delay=46.8ms TP=251.2
...

================================================================================
OVERALL RESULTS (30 files)
================================================================================
DQN Average:
  Acceptance Ratio: 90.45%
  E2E Delay: 48.32ms
  Throughput: 234.56

VAE-DQN Average:
  Acceptance Ratio: 93.78%
  E2E Delay: 44.15ms
  Throughput: 256.78

Improvement:
  AR: +3.33%
  Delay: -8.63%
  Throughput: +9.48%

Plots saved:
  - fig/comparison_grouped.png
  - fig/comparison_by_difficulty.png
  - fig/comparison_by_location.png
Results saved: comparison_results.json
================================================================================
```

**Biểu đồ được tạo:**

1. **comparison_grouped.png**
   - Hiển thị 3 metrics (AR, Delay, Throughput)
   - Nhóm theo `location_difficulty` (VD: atlanta_easy, chicago_medium)
   - So sánh DQN vs VAE-DQN cho từng nhóm

2. **comparison_by_difficulty.png**
   - Hiển thị 3 metrics
   - Nhóm theo mức độ: Easy, Medium, Hard
   - Trung bình tất cả locations cho mỗi mức độ

3. **comparison_by_location.png**
   - Hiển thị 3 metrics
   - Nhóm theo địa danh (Atlanta, Chicago, Dallas, etc.)
   - Trung bình tất cả difficulties cho mỗi location

#### Trên file cụ thể (Optional)
```bash
python main.py compare --data data/cogent_centers_atlanta_easy_s1.json --episodes 20
```

---

## 📊 Chi tiết thành phần

### `agents/`
**DRL agents implementation**

- **`dqn_agent.py`**: 
  - Deep Q-Network với 3 inputs (DC, Demand, Global)
  - Experience replay buffer (50k transitions)
  - Target network update every 10k steps
  - Epsilon-greedy action selection
  
- **`vae_agent.py`**: 
  - VAE Encoder: DC_State → 32D latent
  - VAE Decoder: Latent → Next_DC_State
  - Value Network: Latent → Priority score
  
- **`vae_trainer.py`**: 
  - Circular buffer for VAE data (50k samples)
  - Train VAE with reconstruction + KL loss
  - Train Value Network with MSE loss

### `core/`
**NFV Simulator business logic**

- **`dc.py`**: DataCenter (CPU/RAM/Storage) và SwitchNode
- **`request.py`**: SFC request với VNF chain, bandwidth, deadline
- **`sfc_manager.py`**: Lifecycle management (active/completed/dropped)
- **`simulator.py`**: Discrete-event simulation, time advance
- **`topology.py`**: K-shortest paths, bandwidth allocation/release
- **`vnf.py`**: VNF instance (idle/busy state, processing time)
- **`statistics.py`**: Calculate acceptance ratio, delay, throughput

### `envs/`
**Gym Environment interface**

- **`env.py`**: Main SFCEnvironment class
  - `reset()`: Initialize episode
  - `step(action)`: Execute action, return (state, reward, done, info)
  
- **`action_handler.py`**: 
  - Execute ALLOCATE/UNINSTALL actions
  - Validate resources, bandwidth, latency
  - Calculate rewards
  
- **`observer.py`**: 
  - Construct state representation
  - Padding to MAX_VNF_TYPES=10
  
- **`selectors.py`**: 
  - PrioritySelector: Heuristic DC ordering
  - VAESelector: VAE value-based ordering
  - RandomSelector: Random ordering
  
- **`request_selector.py`**: Priority-based request selection
- **`utils.py`**: Action masking, type parsing

### `runners/`
**Training và testing pipelines**

- **`runner.py`**: Core Runner với các methods
- **`data_generator.py`**: Generate random scenarios
- **`data_loader.py`**: Load from JSON
- **`train_dqn.py`**: DQN training loop
- **`train_vae.py`**: VAE data collection + training
- **`compare.py`**: DQN vs VAE-DQN comparison với grouped analysis

### `config.py`
**Global configuration**
```python
MAX_VNF_TYPES = 10              # Padding size
ACTION_SPACE_SIZE = 21          # 2*10 + 1
MAX_SIM_TIME_PER_EPISODE = 1000 # Max simulation time (ms)
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000
```

## 📝 Input Data Format

### Naming Convention
```
<location>_<difficulty>_s<scenario>.json
```

**Examples:**
- `cogent_centers_atlanta_easy_s1.json`
- `cogent_centers_chicago_medium_s2.json`
- `cogent_centers_dallas_hard_s3.json`

### File Structure
```json
{
  "V": {
    "0": {"server": true, "c_v": 100, "r_v": 200, "h_v": 150, "d_v": 0.1},
    "1": {"server": false}
  },
  "E": [
    {"u": 0, "v": 1, "b_l": 100, "d_l": 0.05}
  ],
  "F": [
    {"c_f": 1.2, "r_f": 1.0, "h_f": 0.8, "d_f": {"0": 0.3}}
  ],
  "R": [
    {"T": 1, "st_r": 0, "d_r": 1, "F_r": [0], "b_r": 1.5, "d_max": 50.0}
  ]
}
```

**Fields:**
- `V`: Nodes (DCs: `server=true`, Switches: `server=false`)
- `E`: Links với bandwidth (`b_l`) và delay (`d_l`)
- `F`: VNF specifications (CPU, RAM, Storage requirements)
- `R`: Requests (arrival time, source, dest, VNF chain, bandwidth, max delay)

## 📈 Performance Metrics

- **Acceptance Ratio**: `completed / (completed + dropped) × 100%`
- **Average E2E Delay**: Mean latency of completed requests
- **Throughput**: Total bandwidth of completed requests

## 🔧 Troubleshooting

### Training quá chậm
```bash
# Giảm số episodes
python main.py train pipeline --episodes 200 --vae-episodes 100

# Hoặc giảm MAX_SIM_TIME_PER_EPISODE trong config.py
MAX_SIM_TIME_PER_EPISODE = 500
```

### Out of memory
```bash
# Giảm MEMORY_SIZE trong config.py
MEMORY_SIZE = 20000
```

### Model không converge
```bash
# Tăng số episodes
python main.py train pipeline --episodes 1000 --vae-episodes 300

# Hoặc điều chỉnh learning rate trong config.py
LEARNING_RATE = 0.0005
```

### Missing plots
```bash
# Cài đặt matplotlib nếu chưa có
pip install matplotlib
```

## 🎓 References

Paper: "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning" (IEEE Networking Letters, 2024)

## 📧 Contact

[Your contact info]