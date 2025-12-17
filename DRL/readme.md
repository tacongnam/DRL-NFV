# DRL-Based SFC Provisioning

Deep Reinforcement Learning approach for Service Function Chaining (SFC) provisioning in NFV networks, based on the paper "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning" (IEEE Networking Letters, 2024).

## 📋 Key Features

- **Reconfigurable Architecture**: DRL model works across different network configurations without retraining
- **Multi-Input DQN**: 3-layer input architecture capturing DC, DC-SFC, and global state
- **Attention Mechanism**: Enhanced neural network with attention layer for better feature learning
- **Priority-Based Scheduling**: Smart DC iteration and VNF allocation based on priority points
- **Resource Reuse**: Efficient VNF instance reuse mechanism
- **Comprehensive Evaluation**: Performance analysis and scalability testing

## 🏗️ Architecture

### State Representation (3 Inputs)
1. **Input 1 - DC State** `[2|V|+2]`: CPU, Storage, Installed VNFs, Idle VNFs
2. **Input 2 - DC-SFC State** `[|S|(1+2|V|)]`: DC-relevant requests, allocated/remaining VNFs
3. **Input 3 - Global State** `[|S|(4+|V|)]`: Request counts, delays, bandwidth, pending VNFs

### Action Space
- **Action 0**: WAIT (do nothing)
- **Actions 1-6**: UNINSTALL VNF types (NAT, FW, VOC, TM, WO, IDPS)
- **Actions 7-12**: ALLOCATE VNF types

### Reward System
- `+2.0`: SFC successfully completed
- `-1.5`: SFC dropped (timeout)
- `-1.0`: Invalid action
- `-0.5`: Uninstall needed VNF
- `0.0`: WAIT action

## 📁 Project Structure

```
├── config.py                  # Configuration parameters
├── scripts.py                 # Main entry point
├── agent/
│   ├── agent.py              # DQN Agent implementation
│   └── model.py              # Neural network architecture
├── environment/
│   ├── gym_env.py            # Gymnasium environment
│   ├── controller.py         # Action execution logic
│   ├── observer.py           # State representation
│   ├── simulator.py          # Time simulation
│   ├── priority.py           # Priority management
│   └── utils.py              # Helper functions
├── spaces/
│   ├── dc.py                 # Data Center class
│   ├── request.py            # SFC Request class
│   ├── vnf.py                # VNF Instance class
│   ├── sfc_manager.py        # Request manager
│   └── topology.py           # Network topology
└── runners/
    ├── train.py              # Training script
    ├── evaluate.py           # Evaluation script
    ├── demo.py               # Demo & validation
    └── utils.py              # Helper functions
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install gymnasium tensorflow numpy matplotlib

# Verify installation
python scripts.py demo
```

### Training

```bash
# Start training (350 updates × 20 episodes)
python scripts.py train

# Results will be saved to:
# - models/sfc_dqn.weights.h5         # Final model
# - fig/training_progress.png         # Training curves
```

### Evaluation

```bash
# Evaluate trained model
python scripts.py eval

# Runs two experiments:
# 1. Performance per SFC type (4 DCs)
# 2. Scalability analysis (2, 4, 6, 8 DCs)
```

### Interactive Mode

```bash
# Launch interactive menu
python scripts.py

# Select:
# 1. Train Model
# 2. Evaluate Model
# 3. Run Demo Tests
# 0. Exit
```

## 🔬 Algorithm Details

### Priority System

**DC Priority (for iteration order)**:
1. **Highest**: Source DC of request with minimum E2E delay
2. **Medium**: DCs on shortest path (closer to source = higher priority)
3. **Lowest**: DCs not on path

**VNF Priority (for allocation)**:
```
P = P1 + P2 + P3

P1 = elapsed_time - max_delay          # Time urgency
P2 = +10 if same DC, -10 if different  # Affinity
P3 = C / (remaining_time + ε)          # Critical urgency
```

### Episode Flow

1. **Start**: Generate initial SFC requests
2. **Loop** (until done):
   - Update DC priority order
   - For each DC (in priority order):
     - Get state representation
     - Select action (ε-greedy)
     - Execute action
     - Collect reward
   - After A actions: advance time (1ms)
   - Generate new requests every N ms
3. **End**: No active requests OR max time reached

### Delay Calculation

**Propagation Delay**:
```
t_prop = distance_ij / speed_of_light
```

**Processing Delay**:
```
t_proc = waiting_time + processing_time
```

**Total E2E Delay**:
```
E2E = Σ(t_prop) + Σ(t_proc)
```

## 📊 SFC Types

| Type | Chain | BW (Mbps) | Delay (ms) | Bundle Size |
|------|-------|-----------|------------|-------------|
| CloudGaming | NAT→FW→VOC→WO→IDPS | 4 | 80 | 40-55 |
| AR | NAT→FW→TM→VOC→IDPS | 100 | 10 | 1-4 |
| VoIP | NAT→FW→TM→FW→NAT | 0.064 | 100 | 100-200 |
| VideoStream | NAT→FW→TM→VOC→IDPS | 4 | 100 | 50-100 |
| MIoT | NAT→FW→IDPS | 1 | 5 | 10-15 |
| Ind4.0 | NAT→FW | 70 | 8 | 1-4 |

## 🎯 Performance Metrics

- **Acceptance Ratio**: % of successfully completed SFCs
- **Drop Ratio**: % of SFCs dropped due to timeout
- **E2E Delay**: Average end-to-end delay for completed SFCs
- **Resource Usage**: CPU/Storage consumption across DCs

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Training parameters
TRAIN_UPDATES = 350              # Total updates
EPISODES_PER_UPDATE = 20         # Episodes per update
ACTIONS_PER_TIME_STEP = 100      # Actions per time step

# DRL parameters
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 50000

# Network parameters
MAX_NUM_DCS = 6
DC_CPU_CYCLES = 12000
DC_RAM = 256
DC_STORAGE = 2048
```

## 📈 Expected Results

Based on the paper:
- **Acceptance Ratio**: ~90% (vs 76% baseline)
- **E2E Delay Reduction**: ~42.65%
- **Resource Reduction**: ~50% storage, ~10% CPU
- **Reconfigurability**: Works across 2-8 DCs without retraining

## 🐛 Troubleshooting

**Issue**: Import errors
```bash
# Solution: Make sure you're in project root
cd /path/to/project
python scripts.py
```

**Issue**: TensorFlow warnings
```bash
# Solution: Already suppressed in code via TF_CPP_MIN_LOG_LEVEL
# If still seeing warnings, they're harmless
```

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size or memory size in config.py
BATCH_SIZE = 32
MEMORY_SIZE = 25000
```

## 📚 References

Paper: "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning"  
Authors: M. A. Onsu, P. Lohan, B. Kantarci, E. Janulewicz, S. Slobodrian  
Published: IEEE Networking Letters, Vol. 6, No. 3, September 2024

## 📝 License

This implementation is for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Advanced priority scheduling algorithms
- Multi-objective optimization (cost, energy)
- Online learning capabilities
- Distributed training support

---

**Note**: First run `demo` to validate setup, then `train` for ~2-4 hours (depends on hardware), finally `eval` to see results.