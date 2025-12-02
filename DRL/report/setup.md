# Setup Guide

## Project Structure

```
DRL-NFV/
├── config.py                 # Configuration parameters
├── utils.py                  # Utility functions
├── dqn_model.py             # DQN with attention layer
├── agent.py                 # DQN agent
├── main.py                  # Main training script
├── demo.py                  # Quick demo
├── debug_test.py            # Debug test suite
├── baseline_heuristic.py    # Baseline comparison
├── tests.py                 # Unit tests
├── requirements.txt         # Dependencies
├── env/
│   ├── __init__.py
│   └── sfc_environment.py   # Gymnasium environment
├── models/                  # Saved models (created automatically)
└── results/                 # Training results (created automatically)
```

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python debug_test.py
```

Expected output: All 6 tests should pass ✓

## Quick Start

### Option 1: Quick Demo (5 minutes)

```bash
python demo.py
```

This runs 5 training episodes and tests on 2, 4, 6 DCs.

### Option 2: Full Training (2-3 hours)

```bash
python main.py
```

This trains for 350 updates × 20 episodes = 7000 episodes.

### Option 3: Run Tests

```bash
# Unit tests
python tests.py -v

# Debug tests  
python debug_test.py
```

## Expected Results

### After Training (main.py):

1. **Console Output:**
   - Training progress per 10 episodes
   - Test results on 2, 4, 6, 8 DCs
   - Baseline comparison
   - Summary statistics

2. **Generated Files:**
   - `models/dqn_sfc.weights.h5` - Trained model weights
   - `training_results.png` - 9 plots visualization

3. **Metrics (vs Baseline):**
   - Acceptance Ratio: +20% improvement
   - E2E Delay: -40% reduction
   - Resource Utilization: -50% reduction

## Troubleshooting

### Issue: TensorFlow warnings

```bash
# Ignore warnings (optional)
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

### Issue: Out of memory

Reduce batch size or memory size in `config.py`:

```python
DRL_CONFIG = {
    'batch_size': 32,        # Default: 64
    'memory_size': 5000,     # Default: 10000
    ...
}
```

### Issue: Training too slow

Reduce training parameters in `config.py`:

```python
DRL_CONFIG = {
    'updates': 100,                  # Default: 350
    'episodes_per_update': 10,       # Default: 20
    ...
}
```

Or run quick demo instead:

```bash
python demo.py
```

### Issue: Import errors

Make sure you're in the project root directory and virtual environment is activated:

```bash
# Check current directory
pwd  # Should show .../DRL-NFV/

# Activate venv if not activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify packages
pip list | grep -E "gymnasium|tensorflow|numpy"
```

## Customization

### Change Network Configuration

In `config.py`:

```python
DC_CONFIG = {
    'cpu_range': (12, 120),  # CPU range for DCs
    'ram': 256,              # RAM in GB
    'storage': 2000,         # Storage in GB
    'link_bw': 1000          # Link bandwidth in Mbps
}
```

### Add New SFC Type

In `config.py`:

```python
SFC_TYPES = {
    'MyNewSFC': {
        'vnfs': ['NAT', 'FW', 'TM'],  # VNF chain
        'bw': 25,                      # Bandwidth (Mbps)
        'delay': 60,                   # E2E delay (ms)
        'bundle': (15, 30)             # Bundle size range
    },
    ...
}
```

### Modify Reward Function

In `config.py`:

```python
REWARD_CONFIG = {
    'sfc_satisfied': 2.0,      # Reward for satisfied SFC
    'sfc_dropped': -1.5,       # Penalty for dropped SFC
    'invalid_action': -1.0,    # Penalty for invalid action
    'uninstall_required': -0.5 # Penalty for wrong uninstall
}
```

## Performance Tuning

### For Faster Training:

1. Reduce episodes: `updates = 100`, `episodes_per_update = 10`
2. Reduce actions per step: `actions_per_step = 30`
3. Use smaller networks: Test on 2-4 DCs only

### For Better Results:

1. Increase training: `updates = 500`, `episodes_per_update = 30`
2. Tune hyperparameters: `learning_rate`, `gamma`, `epsilon_decay`
3. Increase batch size: `batch_size = 128`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{onsu2024unlocking,
  title={Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning},
  author={Onsu, Murat Arda and Lohan, Poonam and Kantarci, Burak and Janulewicz, Emil and Slobodrian, Sergio},
  journal={IEEE Networking Letters},
  volume={6},
  number={3},
  pages={193--197},
  year={2024}
}
```