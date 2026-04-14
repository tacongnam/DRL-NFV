# DRL-NFV

HRL-VGAE pipeline for VNF placement on synthetic NFV topologies.

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

`main.py` supports:

- `generate`
- `pretrain`
- `train`
- `eval`
- `baseline`
- `pipeline`

## Data Generation

```bash
python main.py --mode generate --topology nsf --distribution rural --difficulty easy
```

Key args:

- `--topology`: `nsf`, `conus`, `cogent`
- `--distribution`: `uniform`, `rural`, `urban`, `centers`
- `--difficulty`: `easy`, `hard`
- `--requests`
- `--num-train-files`
- `--num-test-files`

## Pretrain

Only `request_pct` is used to subsample requests.

```bash
python main.py --mode pretrain --train-dir data/train --vgae-epochs 50 --ll-episodes 50 --pretrain-request-pct 10
```

Outputs:

- `models/vgae_pretrained/vgae_weights.npy`
- `models/ll_pretrained/ll_dqn_weights.npy`

## Train

```bash
python main.py --mode train --train-dir data/train --episodes 100 --train-request-pct 10
```

Outputs:

- `models/hrl_final/hl_pmdrl_weights.npy`
- `models/hrl_final/ll_dqn_weights.npy`
- `models/hrl_final/vgae_weights.npy`

Notes:

- training runs across all train files
- checkpoints are saved after each file
- only `train_request_pct` limits request volume

## Eval

```bash
python main.py --mode eval --model-dir models/hrl_final --test-dir data/test
```

Eval now prints progress lines like:

```text
[Eval] 125/2502  acc=...  queue=...  t=...  elapsed=...
```

## Full Pipeline

```bash
python main.py --mode pipeline --episodes 100 --pretrain-request-pct 10 --train-request-pct 10
```

## Baselines

```bash
python main.py --mode baseline --baselines fifs bestfit deadline
```

Available baselines:

- `fifs`
- `glb`
- `spf`
- `bestfit`
- `deadline`
- `random`

## Current Behavior

- pretrain runs inline from `main.py`
- only request percentage is used for dataset limiting
- eval is one-pass and reports progress
- model checkpoints use `.npy`
