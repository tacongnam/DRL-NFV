# HRL-VGAE: VNF Placement trong mạng lõi 5G
**Bài toán:** Đặt VNF trong mạng lõi 5G bằng Hierarchical RL kết hợp VGAE và Pareto-front MO-DRL.

---

## WORKFLOW THỰC HIỆN

### Bước 1: Sinh dữ liệu training

```bash
# Sinh 20 file training (NSF topology, rural distribution, dễ)
python data/generate.py --topology nsf --distribution rural --difficulty easy \
    --num-files 20 --requests 50 --scale 50 --output data/train/

# Sinh file test lớn (COGENT topology, center distribution, khó)
python data/generate.py --topology cogent --distribution centers --difficulty hard \
    --num-files 5 --requests 150 --scale 100 --output data/test/
```

### Bước 2: Pre-train VGAE + LL-DQN (Tùy chọn nhưng khuyến khích)

```bash
python -m models.pretrain --phase both
```

### Bước 3: Training HRL (đa file)

```bash
# Train trên tất cả file trong data/train/
python main.py --mode train --episodes 300 --train-dir data/train/ --model-dir models/hrl_final/

# Train trên file cụ thể
python main.py --mode train --episodes 300 --train-file data/train/nsf_rural_easy_s1.json
```

### Bước 4: Evaluation trên test files

```bash
# Test tất cả file trong data/test/
python main.py --mode eval --model-dir models/hrl_final/ --test-dir data/test/

# Test trên file cụ thể
python main.py --mode eval --model-dir models/hrl_final/ --test-files data/test/COGENT_centers_hard_s1.json
```

---

## THAM SỐ KHUYẾN NGHỊ

| Tham số | Giá trị khuyến nghị | Ghi chú |
|---------|---------------------|---------|
| --episodes | 300-500 | 300: cơ bản, 500+: production |
| --train-dir | data/train/ | Thư mục chứa ~20 file training đã sinh |
| --test-dir | data/test/ | Thư mục chứa file test lớn |
| --ll-pretrained | models/ll_pretrained/... | LL-Score giúp HL chọn tốt hơn |

**Độ tin cậy theo Episodes:** 300 → 60-75%, 500+ → 70-80%

---

## CẤU TRÚC THƯ MỤC
```
├── main.py                 # Entry point (train / eval / baseline)
├── data/
│   ├── generate.py         # Data generator
│   ├── train/              # ~20 training files (sinh bởi generate.py)
│   └── test/               # Test files (sinh bởi generate.py)
├── models/
│   ├── pretrain.py         # Pre-training script
│   ├── vgae_pretrained/    # VGAE weights
│   ├── ll_pretrained/      # LL-DQN weights
│   └── hrl_final/          # Final HRL model (HL + LL weights)
└── strategy/
    ├── hrl.py              # HRL-VGAE strategy
    └── ...
```

## .GITIGNORE
File `.gitignore` đã được cấu hình để:
- Bỏ qua `data/train/`, `data/test/`
- Bỏ qua các file weights `*.h5`
- Giữ lại các file code, config, và ví dụ data