# DRL-NFV

DRL-NFV pipeline for VNF placement on synthetic NFV topologies.

## Requirements

- Python 3.10+
- `numpy`
- `networkx`
- `gymnasium`
- `tensorflow`
- `matplotlib` for plots

Install:

```bash
pip install numpy networkx gymnasium tensorflow matplotlib
```

## Main Modes

`main.py` supports (default: `baseline`):

- `generate` — Generate synthetic NFV scenarios for training/testing
- `pretrain` — Pre-train VGAE + Placer on training data
- `train` — Train DRL agent on training data
- `eval` — Evaluate model on test data
- `baseline` — Run baseline algorithms and compare with DRL (if model exists)

## Generate Data

Generate synthetic NFV test scenarios.

```bash
python main.py --mode generate --topology nsf --distribution rural --difficulty easy --scale 50 --requests 50 --num-test-files 3
```

Args:
- `--topology`: `nsf`, `conus`, `cogent` (default: `nsf`)
- `--distribution`: `uniform`, `rural`, `urban`, `centers` (default: `rural`)
- `--difficulty`: `easy`, `normal`, `hard` (default: `easy`)
- `--scale`: network scale (default: 50)
- `--requests`: requests per scenario (default: 50)
- `--num-test-files`: number of test files to generate (default: 3)

Output:
- Test data: `data/test/*.json`

## Pretrain

Pre-trains VGAE graph encoder and Placer (lower-level policy).

```bash
python main.py --mode pretrain --train-dir data/train --vgae-epochs 60 --ll-episodes 60
```

Args:
- `--vgae-epochs`: VGAE training epochs (default: 60)
- `--ll-episodes`: Placer training episodes (default: 60)
- `--train-dir`: Path to training data (default: `data/train`)

Outputs:
- `models/vgae_pretrained/vgae_weights.npy` — Pre-trained VGAE encoder
- `models/placer/placer_dqn_weights.npy` — Pre-trained Placer policy

## Train

Trains the DRL agent (Placer + Coordinator) with pre-trained components.

```bash
python main.py --mode train --train-dir data/train --episodes 60
```

Args:
- `--episodes`: Training episodes (default: 60)
- `--ll-pretrained`: Path to pre-trained Placer weights (optional, auto-loads if exists)
- `--train-dir`: Path to training data (default: `data/train`)
- `--model-dir`: Output directory for trained model (default: `models/hrl_final`)

Outputs:
- `models/hrl_final/hl_pmdrl_weights.npy` — Trained Coordinator policy
- `models/hrl_final/ll_dqn_weights.npy` — Trained Placer policy
- `models/hrl_final/vgae_weights.npy` — VGAE encoder

Notes:
- Training uses all files in `--train-dir`
- Checkpoints saved after each file

## Eval

Evaluates trained model on test data.

```bash
python main.py --mode eval --model-dir models/hrl_final --test-dir data/test
```

Args:
- `--model-dir`: Path to trained model (default: `models/hrl_final`)
- `--test-dir`: Path to test data (default: `data/test`)
- `--num-runs`: Number of evaluation runs per file (default: 1)

## Baselines

Runs baseline algorithms and optionally compares with DRL model.

```bash
python main.py --mode baseline --test-dir data/test --baselines fifs bestfit deadline randomfit spf glb --plot-out baselines.png
```

Available baselines:
- `fifs` — Greedy First-In-First-Service
- `bestfit` — Best-Fit placement
- `deadline` — Deadline-Aware Greedy
- `randomfit` — Random placement
- `spf` — Shortest Path First
- `glb` — Greedy Load Balancing

Args:
- `--test-dir`: Path to test data (default: `data/test`)
- `--baselines`: Baselines to run (default: all)
- `--model-dir`: Path to DRL model for comparison (default: `models/hrl_final`)
- `--plot-out`: Output path for comparison plot

Outputs:
- CSV file with results (default: `baseline_results.csv`)
- Comparison plots if baseline & DRL models exist

## Quick Start

1. **Generate test data:**

```bash
python main.py --mode generate --num-test-files 5
```

2. **Run baselines on test data:**

```bash
python main.py --mode baseline
```

3. **Evaluate trained model:**

```bash
python main.py --mode eval --test-dir data/test
```

## Typical Workflow

```bash
# 1. Generate test data
python main.py --mode generate --topology nsf --distribution rural --difficulty easy --num-test-files 5

# 2. Pretrain on existing training data
python main.py --mode pretrain --train-dir data/train --vgae-epochs 60 --ll-episodes 60

# 3. Train DRL agent
python main.py --mode train --train-dir data/train --episodes 60

# 4. Evaluate on test data
python main.py --mode eval --test-dir data/test

# 5. Compare with baselines (including all available)
python main.py --mode baseline --test-dir data/test --baselines fifs bestfit deadline randomfit spf glb --plot-out comparison.png
```
