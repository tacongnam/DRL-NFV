# HRL-VGAE: VNF Placement trong mạng lõi 5G
**Bài toán:** Đặt VNF trong mạng lõi 5G bằng Hierarchical RL kết hợp VGAE và Pareto-front MO-DRL.

---

## KIẾN TRÚC HỆ THỐNG

```
┌──────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL AGENT                              │
│  Nhiệm vụ: CHỌN SFC tốt nhất từ Waitlist                         │
│  Input: Z_t (VGAE) + LL_Scores (Pre-trained DQN)                 │
│  Output: SFC* được chọn từ Pareto Front                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LOW-LEVEL AGENT                               │
│  Nhiệm vụ: ĐẶT VNF và ĐỊNH TUYẾN cho SFC* đã chọn                │
│  Input: Z_t (VGAE) + SFC* (đã chọn)                              │
│  Output: Placement Plan (DC placement + Routing paths)           │
└──────────────────────────────────────────────────────────────────┘
```

---

## WORKFLOW THỰC HIỆN

### Bước 1: Sinh dữ liệu training

Dữ liệu training được sinh ngẫu nhiên theo `data/data_configure.txt`:

```bash
# Sinh 10 file training (NSF topology, rural distribution, dễ)
python data/generate.py --topology nsf --distribution rural --difficulty easy \
    --num-files 10 --requests 50 --scale 50 --output data/train/

# Sinh file test lớn (COGENT topology, center distribution, khó)
python data/generate.py --topology cogent --distribution centers --difficulty hard \
    --num-files 5 --requests 100 --scale 100 --output data/test/
```

**Tham số sinh dữ liệu:**

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| --topology | nsf/conus/cogent | Network topology |
| --distribution | uniform/rural/urban/centers | Server node distribution |
| --difficulty | easy/hard | Độ khó (dễ/khó được chấp nhận) |
| --scale | 50-200 | Resource scale (càng lớn càng nhiều resource) |
| --requests | 50-200 | Số request mỗi file |
| --num-files | 1-100 | Số file sinh ra |

### Bước 2: Pre-train VGAE + LL-DQN (Tùy chọn nhưng khuyến khích)

```bash
# Pre-train VGAE
python -m models.pretrain --phase vgae --vgae-epochs 100

# Pre-train LL-DQN (dùng VGAE pretrained nếu có)
python -m models.pretrain --phase ll --ll-episodes 200
```

### Bước 3: Training HRL

```bash
# Train trên file cụ thể
python main.py --mode train --episodes 300 \
    --train-file data/train/nsf_rural_easy_s1.json \
    --ll-pretrained models/ll_pretrained/ll_dqn_weights.weights.h5 \
    --model-dir models/hrl_final/

# Train trên nhiều file (lặp lại command với --train-file khác nhau)
```

**Tham số training:**

| Tham số | Giá trị khuyến nghị | Ghi chú |
|---------|---------------------|---------|
| --episodes | 300-500 | 300: cơ bản, 500+: production |
| --ll-pretrained | models/ll_pretrained/... | LL-Score giúp HL chọn tốt hơn |

### Bước 4: Evaluation

```bash
# Test trên thư mục chứa file test
python main.py --mode eval \
    --model-dir models/hrl_final/ \
    --test-dir data/test/

# Test trên file cụ thể
python main.py --mode eval \
    --model-dir models/hrl_final/ \
    --test-files data/test/cogent_centers_hard_s1.json data/test/cogent_centers_hard_s2.json
```

---

## THAM SỐ KHUYẾN NGHỊ

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Episodes | **300 - 500** | 300: cơ bản, 500+: stable hơn |
| Learning Rate | **0.0005** | Standard for DRL |
| Gamma (γ) | **0.95** | Discount factor |
| Batch Size | **32** | Standard |
| Epsilon decay | 1.0 → 0.01 | Exponential |
| Target update | **500 steps** | Stabilizes training |

**Độ tin cậy theo Episodes:**

| Episodes | Dự kiến Acceptance Ratio |
|----------|-------------------------|
| 50 - 100 | 20% - 40% |
| 100 - 200 | 40% - 60% |
| **300** | **60% - 75%** |
| 500+ | 70% - 80% |

---

## CẤU TRÚC THƯ MỤC

```
├── main.py                 # Entry point (train / eval / baseline)
├── config.py               # Cấu hình hệ thống
├── data/
│   ├── generate.py         # Data generator
│   ├── data_configure.txt  # Data format specification
│   ├── train/              # Training data (sinh bởi generate.py)
│   └── test/               # Test data (sinh bởi generate.py)
├── models/
│   ├── model.py            # Neural network definitions
│   ├── pretrain.py         # Pre-training script
│   ├── vgae_pretrained/    # VGAE weights
│   ├── ll_pretrained/      # LL-DQN weights
│   └── hrl_final/          # Final HRL model (HL + LL weights)
└── strategy/
    ├── hrl.py              # HRL-VGAE strategy
    ├── fifs.py             # Baseline: First-In-First-Serve
    └── glb.py              # Baseline: Greedy Load Balance
```

---

## ĐỊNH DẠNG DỮ LIỆU JSON

Theo `data/data_configure.txt`:

```json
{
  "V": {"v0": {"server": true, "c_v": 100, "r_v": 64, "h_v": 100, ...}},
  "E": [{"u": "v0", "v": "v1", "b_l": 40.0, "d_l": 1.0}],
  "F": [{"c_f": 10, "r_f": 8, "h_f": 20, "d_f": {}}],
  "R": [{"T": 0, "st_r": "v0", "d_r": "v5", "F_r": [0, 1, 2], "b_r": 1.0, "d_max": 10}]
}