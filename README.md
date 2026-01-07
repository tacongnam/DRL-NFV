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
├── data/                            # Test datasets (30 files)
│   ├── scenario_001.json
│   ├── scenario_002.json
│   └── ...
│
├── models/                          # Saved models
│   ├── best_model_q.weights.h5    # DQN model
│   ├── vae_model_*.weights.h5      # VAE models
│   └── checkpoint_*.weights.h5     # Checkpoints
│
├── fig/                             # Output figures
│   └── comparison.png              # DQN vs VAE-DQN comparison
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

### Step 1: Train DQN (Random Data)
```bash
python main.py train random --episodes 500
```

**Chức năng:**
- Generate random scenarios mỗi episode với progressive difficulty:
  - Episode 0-30%: DC=4-6, Nodes=10-16, Requests=15-30 (Easy)
  - Episode 30-60%: DC=5-8, Nodes=15-23, Requests=30-50 (Medium)
  - Episode 60-100%: DC=6-10, Nodes=16-30, Requests=40-80 (Hard)
- Epsilon decay: 1.0 → 0.01
- Checkpoint every 50 episodes
- Save best model to `models/best_model`

**Output mẫu:**
```
Episode 1/500 [DC:4 N:14 VNF:6 Req:25]: R=450 AR=76.0% C:19 D:6 S:3245
Episode 2/500 [DC:6 N:18 VNF:8 Req:35]: R=623 AR=82.9% C:29 D:6 S:4521
...
Checkpoint 50: AR=85.34% R=712.3 Memory=45230
...
TRAINING COMPLETE: AR=94.28% R=1523.5
```

**Train với file cụ thể (optional):**
```bash
python main.py train dqn --data data/scenario_001.json --updates 40
```

---

### Step 2: Train VAE (Random Data)
```bash
python main.py train vae --episodes 200
```

**Chức năng:**
- Collect DC state transitions từ random scenarios
- Train VAE Encoder + Decoder để predict next DC state
- Train Value Network để score DC priority
- Save to `models/vae_model`

**Output mẫu:**
```
Collecting VAE data: 200 episodes

Episode 10/200: 45230 samples
Episode 20/200: 89450 samples
...
Collected 234567 transitions

>>> Training VAE (234567 samples)...
    Epoch 5/50 - Loss: 0.4523
    Epoch 10/50 - Loss: 0.3241
    ...

>>> Training Value Network (234567 samples)...
    Epoch 5/100 - Loss: 0.2134
    Epoch 10/100 - Loss: 0.1567
    ...

✓ VAE model saved to models/vae_model
```

**Train với file cụ thể (optional):**
```bash
python main.py train vae --data data/scenario_001.json --vae-episodes 100
```

---

### Step 3: Compare DQN vs VAE-DQN
```bash
python main.py compare
```

**Chức năng:**
- Load cả 2 models (DQN và VAE-DQN)
- Test trên **TẤT CẢ** files trong `data/` (30 files)
- So sánh performance: Acceptance Ratio, E2E Delay
- Vẽ biểu đồ comparison

**Output:**
```
================================================================================
Comparing DQN vs VAE-DQN on all files in data/
================================================================================

Testing 30 files...

File 1/30: scenario_001.json
  DQN:     AR=92.3% Delay=47.5ms
  VAE-DQN: AR=95.1% Delay=43.2ms

File 2/30: scenario_002.json
  DQN:     AR=89.7% Delay=51.3ms
  VAE-DQN: AR=93.4% Delay=46.8ms
...

================================================================================
OVERALL RESULTS (30 files)
================================================================================
DQN:
  Avg Acceptance Ratio: 90.45% ± 3.21%
  Avg E2E Delay: 48.32ms ± 5.67ms

VAE-DQN:
  Avg Acceptance Ratio: 93.78% ± 2.89%
  Avg E2E Delay: 44.15ms ± 4.23ms

Improvement:
  Acceptance Ratio: +3.33%
  E2E Delay: -8.63%

Plot saved: fig/comparison.png
Results saved: test_results.json
================================================================================
```

**Compare với file cụ thể (optional):**
```bash
python main.py compare --data data/scenario_001.json --episodes 20
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
- **`compare.py`**: DQN vs VAE-DQN comparison

### `config.py`
**Global configuration**
```python
MAX_VNF_TYPES = 10              # Padding size
ACTION_SPACE_SIZE = 21          # 2*10 + 1
MAX_SIM_TIME_PER_EPISODE = 5000 # Max simulation time (ms)
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000
```

## 📝 Input Data Format
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
python main.py train random --episodes 200

# Hoặc giảm MAX_SIM_TIME_PER_EPISODE trong config.py
MAX_SIM_TIME_PER_EPISODE = 3000
```

### Out of memory
```bash
# Giảm MEMORY_SIZE trong config.py
MEMORY_SIZE = 20000
```

### Model không converge
```bash
# Tăng số episodes
python main.py train random --episodes 1000

# Hoặc điều chỉnh learning rate trong config.py
LEARNING_RATE = 0.0005
```

## 🎓 References

Paper: "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning" (IEEE Networking Letters, 2024)

## 📧 Contact

[Your contact info]