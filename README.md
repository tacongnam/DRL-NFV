# DRL-NFV: Deep Reinforcement Learning for NFV VNF Placement

A Hierarchical Reinforcement Learning approach with Variational Graph Autoencoder (HRL-VGAE) for optimizing Virtual Network Function (VNF) placement in Network Function Virtualization (NFV) environments.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Generate   │ ──▶ │  Pre-train   │ ──▶ │    Train     │ ──▶ │    Eval      │
│  (data/)    │     │  (VGAE+LL)   │     │   (HRL)      │     │  (test/)     │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Project Structure

```
DRL-NFV/
├── main.py                 # Main entry point with all modes
├── config.py               # Configuration constants
├── data/
│   ├── generate.py         # Synthetic data generator
│   ├── train/              # Training data (generated)
│   └── test/               # Test data (generated)
├── models/
│   ├── model.py            # Neural network architectures
│   ├── pretrain.py         # Pre-training script (VGAE + LL-DQN)
│   ├── train.py            # Legacy training script
│   ├── vgae_pretrained/    # Pre-trained VGAE weights
│   └── ll_pretrained/      # Pre-trained Low-Level DQN weights
├── env/                    # Environment modules
│   ├── env.py              # Main environment
│   ├── network.py          # Network topology
│   ├── request.py          # Request/SFC handling
│   └── vnf.py              # VNF definitions
└── strategy/
    ├── hrl.py              # HRL-VGAE strategy (main algorithm)
    ├── fifs.py             # Greedy FIFS baseline
    └── glb.py              # Greedy GLB baseline
```

## Quick Start

### Full Automated Pipeline

Run the complete pipeline from data generation to evaluation:

```bash
python main.py --mode pipeline \
               --topology nsf \
               --distribution rural \
               --difficulty easy \
               --num-train-files 5 \
               --num-test-files 3 \
               --episodes 300
```

### Step-by-Step Execution

#### Step 1: Generate Training/Test Data

```bash
# Generate data using main.py
python main.py --mode generate \
               --topology nsf \
               --distribution rural \
               --difficulty easy \
               --num-train-files 10 \
               --num-test-files 3

# Or use generate.py directly
python data/generate.py --topology nsf --distribution rural \
                        --difficulty easy --num-files 10 \
                        --scale 50 --output data/train
```

#### Step 2: Pre-train Models

Pre-train VGAE (graph encoder) and Low-Level DQN (VNF placement):

```bash
# Using main.py
python main.py --mode pretrain --train-dir data/train

# Or use pretrain.py directly
python models/pretrain.py --phase both --train-dir data/train \
                          --vgae-epochs 100 --ll-episodes 200
```

#### Step 3: Train HRL Policy

Train the High-Level RL agent (SFC scheduling):

```bash
python main.py --mode train --episodes 300 \
               --ll-pretrained models/ll_pretrained \
               --train-dir data/train
```

#### Step 4: Evaluate on Test Data

```bash
python main.py --mode eval --model-dir models/hrl_final \
               --test-dir data/test
```

### Run Baselines

Compare with greedy baselines:

```bash
python main.py --mode baseline
```

## Pipeline Modes

| Mode | Description |
|------|-------------|
| `pipeline` | Full automated: generate → pretrain → train → eval |
| `generate` | Generate synthetic training/test data |
| `pretrain` | Pre-train VGAE and Low-Level DQN |
| `train` | Train HRL policy |
| `eval` | Evaluate trained model on test data |
| `baseline` | Run greedy baselines (FIFS, GLB) |

## Command-Line Arguments

### Data Generation
- `--topology`: Network topology (`nsf`, `conus`, `cogent`)
- `--distribution`: Server distribution (`uniform`, `rural`, `urban`, `centers`)
- `--difficulty`: Difficulty level (`easy`, `hard`)
- `--scale`: Resource scale factor (default: 50)
- `--requests`: Number of requests per file (default: 50)
- `--num-train-files`: Number of training files to generate
- `--num-test-files`: Number of test files to generate

### Training
- `--episodes`: Training episodes (default: 300)
- `--ll-pretrained`: Path to pre-trained LL model
- `--model-dir`: Directory to save/load models
- `--train-dir`: Directory containing training files

### Pre-training
- `--vgae-epochs`: VGAE pre-training epochs (default: 100)
- `--ll-episodes`: LL-DQN pre-training episodes (default: 200)

## Algorithm: HRL-VGAE

The Hierarchical Reinforcement Learning with Variational Graph Autoencoder consists of:

1. **VGAE (Graph Encoder)**: Encodes network topology into latent node embeddings
2. **Low-Level Agent (LL-DQN)**: Places individual VNFs onto data center nodes
3. **High-Level Agent (HL-PM-DRL)**: Schedules SFCs using Pareto-based Multi-Objective DRL

### Pre-training Flow

1. **VGAE Pre-training**: Collect graph snapshots from training environments, train VGAE to reconstruct adjacency matrices
2. **LL-DQN Pre-training**: Use pre-trained VGAE embeddings to train VNF placement decisions

### Training Flow

1. **State Representation**: VGAE encodes current network state into latent embeddings
2. **High-Level Action**: HL agent selects which SFC to process next
3. **Low-Level Action**: LL agent places each VNF of the selected SFC
4. **Reward**: Based on acceptance ratio, resource utilization, and delay constraints
5. **Update**: Both HL and LL agents update their policies via experience replay

## Dependencies

- TensorFlow 2.x
- NumPy
- NetworkX
- Python 3.7+

```bash
pip install tensorflow numpy networkx
```

## Files Output

After running the pipeline:

```
models/
├── vgae_pretrained/
│   └── vgae_weights.weights.h5       # Pre-trained VGAE
├── ll_pretrained/
│   └── ll_dqn_weights.weights.h5      # Pre-trained Low-Level DQN
└── hrl_final/
    ├── ll_dqn_weights.weights.h5       # Final Low-Level model
    └── hl_pmdrl_weights.weights.h5     # Final High-Level model
```

## Citation

If you use this code, please cite the relevant paper.

## License

MIT License