# DRL-NFV: Baseline Strategies for VNF Placement

This repository contains baseline heuristic strategies for the **Service Function Chaining (SFC) placement** problem in Network Function Virtualization (NFV) environments.

## Strategies

| Strategy | Key Idea |
|---|---|
| **GreedyFIFS** | First-In-First-Served: processes requests by arrival time, places VNFs on lowest-cost DCs |
| **BestFit** | Bin-packing style: places VNFs on DCs with minimum remaining resources (least waste) |
| **DeadlineAwareGreedy** | Earliest-deadline-first: prioritizes tight-deadline requests, prefers nearby DCs when time is critical |

Additional strategies in the repo: `GreedyGLB`, `ShortestPathFirst`, `RandomFit`, and `HRL-VGAE` (Deep RL).

---

## Project Structure

```
DRL-NFV/
├── config.py                   # Constants (resource types, timestep)
├── main.py                     # CLI entry point
├── env/
│   ├── vnf.py                  # VNF type definitions
│   ├── request.py              # Request and SFC classes
│   ├── network.py              # Node, Link, Network classes
│   └── env.py                  # Environment + Strategy base class
├── strategy/
│   ├── fifs.py                 # GreedyFIFS
│   ├── best_fit.py             # BestFit
│   ├── deadline_aware.py       # DeadlineAwareGreedy
│   ├── glb.py                  # GreedyGLB (Global Load Balancing)
│   ├── spf.py                  # ShortestPathFirst
│   ├── random_fit.py           # RandomFit
│   └── hrl.py                  # HRL-VGAE (requires TensorFlow)
├── data/
│   ├── generate.py             # Synthetic data generator
│   ├── data_configure.txt      # Data format documentation
│   ├── train/                  # Training data (generated)
│   └── test/                   # Test data (pre-generated)
├── models/                     # RL model code and weights
├── kaggle_baselines.ipynb      # Self-contained Kaggle notebook
└── README.md                   # This file
```

---

## Option A: Run Offline (Local Machine / Server)

### Prerequisites

- Python >= 3.10
- Required packages: `networkx`, `gymnasium`, `matplotlib`, `pandas`, `numpy`

### 1. Install Dependencies

```bash
pip install networkx gymnasium matplotlib pandas numpy
```

> **Note:** The HRL-VGAE strategy additionally requires `tensorflow`. The baseline strategies do NOT need TensorFlow.

### 2. Generate Data (optional — test data is already included)

```bash
# Generate training data
python data/generate.py --topology nsf --distribution rural --difficulty easy \
    --scale 50 --num-files 5 --requests 50 --output data/train

# Generate test data
python data/generate.py --topology nsf --distribution rural --difficulty easy \
    --scale 50 --num-files 3 --requests 50 --output data/test
```

**Topology options:** `nsf` (14 nodes), `conus` (75 nodes), `cogent` (197 nodes)
**Distribution options:** `uniform`, `rural`, `urban`, `centers`
**Difficulty options:** `easy`, `hard`

### 3. Run Baseline Strategies

```bash
# Run all 6 baselines (excluding HRL which needs TensorFlow)
python main.py --mode baseline --baselines fifs bestfit deadline glb spf random

# Run only the 3 key baselines
python main.py --mode baseline --baselines fifs bestfit deadline

# Specify test data directory
python main.py --mode baseline --baselines fifs bestfit deadline --test-dir data/test

# Save comparison plot
python main.py --mode baseline --baselines fifs bestfit deadline --plot-out results.png
```

### 4. Run HRL-VGAE (Full Pipeline)

```bash
# Install TensorFlow first
pip install tensorflow

# Full pipeline: generate -> pretrain VGAE -> train HRL -> evaluate
python main.py --mode pipeline --topology nsf --distribution rural \
    --difficulty easy --episodes 300

# Or run each step individually:
python main.py --mode generate --topology nsf --distribution rural --difficulty easy
python main.py --mode pretrain
python main.py --mode train --episodes 300
python main.py --mode eval
```

---

## Option B: Run on Kaggle

### Quick Start

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Click **File > Import Notebook**
4. Upload `kaggle_baselines.ipynb` from this repo
5. Click **"Run All"** — the notebook is fully self-contained

### What the Notebook Does

The notebook bundles all source code into a single file:

- **Cell 1:** Install dependencies (`networkx`, `gymnasium`, etc.)
- **Cells 2-6:** Define all core modules (config, VNF, Network, Env, Strategies)
- **Cell 7:** Data generator (generates synthetic NFV data in-memory)
- **Cells 8-9:** Run all 3 strategies and print statistics
- **Cell 10:** Summary table (pandas DataFrame)
- **Cell 11:** Bar chart comparison (acceptance ratio, cost, delay)
- **Cell 12:** Multi-scenario benchmark across topologies and distributions

### Changing Parameters

In the experiment parameters cell, modify these variables:

```python
TOPOLOGY     = "nsf"       # "nsf" (14 nodes), "conus" (75), "cogent" (197)
DISTRIBUTION = "rural"     # "uniform", "rural", "urban", "centers"
DIFFICULTY   = "easy"      # "easy" or "hard"
SCALE        = 50          # resource scale factor
NUM_REQUESTS = 200         # number of SFC requests
```

> **Tip:** For Kaggle's free tier, stick with `nsf` topology (14 nodes, runs in seconds). `conus` (75 nodes) takes ~1 min. `cogent` (197 nodes) may take several minutes.

---

## Metrics

| Metric | Description |
|---|---|
| **Acceptance Ratio** | Fraction of requests successfully placed |
| **Total Cost** | Sum of deployment costs for all accepted requests |
| **Total Delay** | Sum of (end_time - arrival_time) for accepted requests |
| **Average Cost** | Total cost / number of accepted requests |

---

## Data Format

Each JSON data file contains:

```json
{
  "V": { "node_name": {"server": true, "c_v": 200, "r_v": 128, "h_v": 200, ...} },
  "E": [ {"u": "v0", "v": "v1", "b_l": 40.0, "d_l": 1.0} ],
  "F": [ {"c_f": 5, "r_f": 4, "h_f": 10, "d_f": {}} ],
  "R": [ {"T": 0.5, "st_r": "v0", "d_r": "v3", "F_r": [0,1,2], "b_r": 1.0, "d_max": 15} ]
}
```

See `data/data_configure.txt` for full documentation.
