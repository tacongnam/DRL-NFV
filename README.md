# DRL-NFV: Deep Reinforcement Learning for NFV VNF Placement

Hierarchical Reinforcement Learning with Variational Graph Autoencoder (HRL-VGAE) for optimizing Virtual Network Function (VNF) placement in Network Function Virtualization (NFV) environments.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Generate   │ ──▶ │  Pre-train   │──▶ │    Train     │──▶ │    Eval      │
│  (data/)    │     │  (VGAE+LL)   │     │   (HRL)      │     │  (test/)     │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Project Structure

```
DRL-NFV/
├── main.py                     # Entry point
├── config.py                   # Constants
├── data/
│   ├── generate.py             # Synthetic data generator
│   ├── train/                  # Training data
│   └── test/                   # Test data
├── models/
│   ├── model.py                # VGAE, HighLevelAgent, LowLevelAgent, ReplayBuffer
│   ├── pretrain.py             # Phase 1: VGAE + Phase 2: LL-DQN imitation
│   └── train.py                # HRL training entry point
├── env/
│   ├── env.py                  # Gym environment + Strategy base class
│   ├── network.py              # Node, Link, Network (event-driven resource tracking)
│   ├── request.py              # Request, SFC, ListOfRequests
│   └── vnf.py                  # VNF, ListOfVnfs
└── strategy/
    ├── hrl.py                  # HRL-VGAE (main algorithm)
    ├── fifs.py                 # Greedy FIFS baseline
    ├── glb.py                  # Greedy GLB baseline
    ├── spf.py                  # Shortest Path First baseline
    ├── best_fit.py             # Best Fit (minimize waste) baseline
    ├── deadline_aware.py       # Deadline-Aware Greedy baseline
    └── random_fit.py           # Random Fit baseline (lower bound)
```

## Algorithm: HRL-VGAE

### Overview

Three-component hierarchical architecture:

1. **VGAE (Variational Graph Autoencoder)**: Encodes the DC-level network topology into latent node embeddings Z. Pre-trained unsupervised on graph reconstruction; fine-tuned online during HRL training.

2. **High-Level Agent (HL-PMDRL)**: Pareto-front Multi-Objective DRL. Given the current queue of pending SFCs and the VGAE embedding Z, selects which SFC to process next. Outputs two Q-values per SFC: Q^AR (acceptance rate objective) and Q^Cost (cost objective). Non-dominated sorting selects the Pareto-optimal SFC.

3. **Low-Level Agent (LL-DQN)**: Given Z and the VNF resource vector, selects which DC to place the current VNF on. Uses action masking to restrict choices to feasible DCs. An adaptive weight network (trained jointly) learns α and β for the shaped reward.

### Pre-training

**Phase 1 – VGAE**: Collect DC-graph snapshots from training environments. Train VGAE via ELBO (BCE reconstruction + β·KL). Output: `models/vgae_pretrained/vgae_weights.npy`.

**Phase 2 – LL-DQN imitation**: Run GreedyFIFS as teacher. For each VNF, push positive sample (greedy choice, reward=+2.0) and up to 3 negative samples (non-greedy DCs, reward=-1.0). Train LL-DQN via experience replay. Output: `models/ll_pretrained/ll_dqn_weights.weights.h5`.

### Training Loop (per episode)

```
for each SFC (sorted by arrival_time):
    1. Clear graph cache
    2. Encode network → Z via VGAE
    3. HL selects SFC (ε-greedy over Pareto front)
    4. With prob greedy_prob: use GreedyFIFS (imitation); else: use LL-DQN
    5. env.step(plan) → success/fail
    6. Compute R_HL = [±BaseAR, -cost_norm], R_LL = shaped reward
    7. Push to buf_HL, buf_LL, buf_Graph
    8. Train HL, LL, VGAE from buffers
    9. Sync target networks every TARGET_SYNC steps
```

**ε-decay**: `ε = 0.01 + 0.99 · exp(-3·step / total_steps)`

**Greedy curriculum**: `greedy_prob = max(0, 1 - step / (0.8 · total_steps))` — starts near 1 (pure imitation), decays to 0 (pure RL).

**Rollback**: On failure, restore network snapshot (lightweight dict copy of `.used` fields only, not full deepcopy).

### Evaluation (real-time)

Event-driven queue: instead of stepping by TIMESTEP, advance time to the next arrival or deadline event. This reduces eval complexity from O(T/Δt) to O(|events|).

```
while pending or queue:
    admit arriving SFCs
    drop expired SFCs
    HL selects best SFC via Pareto front (ε=0)
    LL places VNFs (ε=0), fallback to GreedyFIFS on failure
    on success: advance to next arrival if queue empty
    on failure: re-queue SFC, jump to next event
```

### Resource Tracking (event-driven)

`Node.used` and `Link.used` are sparse dicts `{timeslot: usage}`. Resource checks and availability queries only inspect keys within the queried interval plus the nearest preceding key as baseline. This avoids iterating over thousands of timeslots.

## Baseline Algorithms

| Key | Class | Strategy |
|-----|-------|----------|
| `fifs` | GreedyFIFS | Minimum cost DC, FIFO order |
| `glb` | GreedyGLB | Maximum remaining resource ratio (load balance) |
| `spf` | ShortestPathFirst | Minimum routing delay to DC |
| `bestfit` | BestFit | Minimum resource waste (pack tightly) |
| `deadline` | DeadlineAwareGreedy | Delay-aware if deadline tight, else min-cost; EDF ordering |
| `random` | RandomFit | Random DC selection (lower bound reference) |

## Quick Start

### Full Pipeline

```bash
python main.py --mode pipeline \
               --topology nsf \
               --distribution rural \
               --difficulty easy \
               --num-train-files 5 \
               --num-test-files 3 \
               --episodes 300
```

### Step by Step

```bash
# 1. Generate data
python main.py --mode generate --topology nsf --distribution rural \
               --difficulty easy --num-train-files 10 --num-test-files 3

# 2. Pre-train
python main.py --mode pretrain --train-dir data/train \
               --vgae-epochs 200 --ll-episodes 200

# 3. Train HRL
python main.py --mode train --episodes 300 --train-dir data/train

# 4. Evaluate
python main.py --mode eval --model-dir models/hrl_final --test-dir data/test

# 5. Run all baselines + plot
python main.py --mode baseline --plot-out results/baseline.png

# 5b. Run specific baselines only
python main.py --mode baseline --baselines fifs glb spf
```

### Baseline Mode

When `--mode baseline` is run:
- Runs all (or selected) baseline algorithms on the first test file
- Prints comparison table
- Generates a 3-panel bar chart (Acceptance Ratio, Total Cost, Total Delay)
- If a trained HRL model exists in `--model-dir`, runs HRL evaluation and generates a second comparison chart (HRL vs baselines)

```bash
python main.py --mode baseline \
               --test-dir data/test \
               --model-dir models/hrl_final \
               --plot-out results/comparison.png
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `baseline` | Run mode |
| `--topology` | `nsf` | Network topology (`nsf`, `conus`, `cogent`) |
| `--distribution` | `rural` | Server distribution |
| `--difficulty` | `easy` | Request difficulty |
| `--scale` | `50` | Resource scale factor |
| `--requests` | `50` | Requests per file |
| `--num-train-files` | `5` | Training files to generate |
| `--num-test-files` | `3` | Test files to generate |
| `--episodes` | `300` | Training episodes |
| `--ll-pretrained` | auto | Path to pre-trained LL weights |
| `--model-dir` | `models/hrl_final` | Model save/load directory |
| `--train-dir` | `data/train` | Training data directory |
| `--test-dir` | `data/test` | Test data directory |
| `--vgae-epochs` | `200` | VGAE pre-training epochs |
| `--ll-episodes` | `200` | LL-DQN pre-training episodes |
| `--baselines` | all | Subset of baselines to run |
| `--plot-out` | screen | Save plot to file path |

## Dependencies

```bash
pip install tensorflow numpy networkx matplotlib gymnasium
```

- Python 3.8+
- TensorFlow 2.x
- NetworkX
- Matplotlib (for baseline plots)
- Gymnasium

## Model Files

```
models/
├── vgae_pretrained/
│   └── vgae_weights.npy
├── ll_pretrained/
│   └── ll_dqn_weights.weights.h5
└── hrl_final/
    ├── ll_dqn_weights.weights.h5
    ├── hl_pmdrl_weights.weights.h5
    └── vgae_weights.npy
```

## Known Limitations

- `TIMESTEP = 0.01` in `config.py` determines timeslot granularity. Resource tracking is event-driven so this only affects `_get_timeslot()` integer conversion.
- HRL evaluation is event-driven (fast). HRL training processes SFCs sequentially per episode (not timeslot-stepped) for speed.
- `conus` and `cogent` topologies generate random link structures; results may vary across runs without fixed seeds.