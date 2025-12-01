# DQN Architecture - Chi tiết Implementation

## 1. Kiến trúc Model (Theo Paper)

### Input Layer Structure
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   State 1   │  │   State 2   │  │   State 3   │
│  (DC Info)  │  │ (SFC@DC)    │  │(Overall SFC)│
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                 │                 │
       ▼                 ▼                 ▼
   ┌───────┐        ┌───────┐        ┌───────┐
   │FC-128 │        │FC-128 │        │FC-128 │
   │  +BN  │        │  +BN  │        │  +BN  │
   └───┬───┘        └───┬───┘        └───┬───┘
       │                 │                 │
   ┌───▼───┐        ┌───▼───┐        ┌───▼───┐
   │FC-128 │        │FC-128 │        │FC-128 │
   │ ReLU  │        │ ReLU  │        │ ReLU  │
   └───┬───┘        └───┬───┘        └───┬───┘
       │                 │                 │
   ┌───▼───┐        ┌───▼───┐        ┌───▼───┐
   │ FC-64 │        │ FC-64 │        │ FC-64 │
   │ ReLU  │        │ ReLU  │        │ ReLU  │
   └───┬───┘        └───┬───┘        └───┬───┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
                ▼                 ▼
           ┌────────────────────────┐
           │    Concatenation       │
           │     (64+64+64=192)     │
           └───────────┬────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   FC-192 ReLU   │
              └────────┬─────────┘
                       │
                       ▼
         ┌──────────────────────────┐
         │   Attention Mechanism    │
         │  W1, W2, V (Bahdanau)    │
         │   α = softmax(V⋅tanh())  │
         │   context = α ⊗ input    │
         └──────────┬───────────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Flatten    │
            └──────┬───────┘
                   │
                   ▼
         ┌─────────────────┐
         │  FC-256 + BN    │
         │  ReLU + Drop0.3 │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  FC-256 + BN    │
         │  ReLU + Drop0.2 │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  FC-128 + BN    │
         │      ReLU       │
         └────────┬─────────┘
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
    ┌──────────┐    ┌──────────┐
    │ Value    │    │Advantage │
    │ Stream   │    │ Stream   │
    │  FC-64   │    │  FC-64   │
    │   ↓      │    │   ↓      │
    │  FC-1    │    │ FC-|A|   │
    └────┬─────┘    └────┬─────┘
         │               │
         └───────┬───────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  Q(s,a) = V(s) +          │
    │  [A(s,a) - mean(A(s,⋅))]  │
    │   (Dueling DQN)           │
    └───────────┬────────────────┘
                │
                ▼
          ┌──────────┐
          │ Q-values │
          │  (|A|)   │
          └──────────┘
```

## 2. State Representation (3-Layer Input)

### State 1: Current DC Information
- Dimension: `[1 × (2×|V| + 2)]` = `[1 × 14]`
- Content:
  ```
  [installed_NAT, avail_NAT, installed_FW, avail_FW, ..., 
   storage_avail, cpu_avail]
  ```
- Normalized: [0, 1]

### State 2: SFC Processing at Current DC
- Dimension: `[|S| × (1 + 2×|V|)]` = `[6 × 13]`
- Content per SFC type:
  ```
  [sfc_type_id, allocated_NAT, allocated_FW, ..., 
   waiting_NAT, waiting_FW, ...]
  ```
- Shows which VNFs are allocated/waiting for each SFC type

### State 3: Overall Pending SFC Requests
- Dimension: `[|S| × (4 + |V|)]` = `[6 × 10]`
- Content per SFC type:
  ```
  [sfc_type_id, pending_count, min_remaining_delay, 
   total_bw, waiting_NAT, waiting_FW, ...]
  ```
- Provides global view of all pending requests

## 3. Action Space

Total actions: `2×|V| + 1 = 13`

```
Actions:
├─ 0-5:   Allocate VNF [NAT, FW, VOC, TM, WO, IDPS]
├─ 6-11:  Uninstall VNF [NAT, FW, VOC, TM, WO, IDPS]
└─ 12:    Wait (no operation)
```

## 4. Attention Mechanism (Bahdanau-style)

```python
score = V · tanh(W1·concat + W2·concat)
attention_weights = softmax(score)
context = attention_weights ⊗ concat
```

**Mục đích**: Highlight các features quan trọng trong concatenated state

## 5. Q-Learning Update (DQN với Target Network)

### Q-value Calculation:
```
Q_target(s,a) = r + γ × max_a' Q_target(s', a')  if not done
Q_target(s,a) = r                                 if done
```

### Loss Function:
```
L = Huber_Loss(Q_target, Q_predicted)
```

### Training Steps:
1. Sample batch từ Replay Memory
2. Calculate target Q-values using **target network**
3. Calculate predicted Q-values using **main network**
4. Compute loss và backprop
5. Update target network mỗi 10 updates

## 6. Dueling DQN

**Formula**:
```
Q(s,a) = V(s) + [A(s,a) - mean_a(A(s,a))]
```

**Ưu điểm**:
- V(s): Đánh giá state tốt hay xấu
- A(s,a): Advantage của action này so với trung bình
- Tách biệt → học tốt hơn cho states có nhiều actions tương đương

## 7. Priority-based VNF Selection

Khi có nhiều VNF cùng type cần allocate:

```
Priority = P1 + P2 + P3

P1 = elapsed_time
P2 = DC_priority (±10 tùy VNFs trước đã allocate ở đâu)
P3 = urgency_boost = C/(remaining_time + ε) if remaining < threshold
```

## 8. Reconfigurability

**Vì sao model hoạt động với bất kỳ NC nào?**

✓ State dimensions độc lập với số DC:
  - State chỉ mô tả DC hiện tại, không phụ thuộc tổng số DC
  - Action space cố định (13 actions)
  
✓ DC iteration theo priority:
  - Model traverse qua DCs theo thứ tự ưu tiên
  - Không hardcode số lượng DC

✓ Flexible path finding:
  - Shortest path calculation động
  - Bandwidth allocation runtime

## 9. Training Configuration

```yaml
Updates: 350
Episodes per update: 20
Actions per step: 100
Epsilon decay: 1.0 → 0.01 (decay=0.995)
Gamma: 0.95
Learning rate: 0.0001
Batch size: 64
Memory size: 100,000
Target update frequency: 10
```

## 10. Performance Targets (vs Baseline)

| Metric | Target Improvement |
|--------|-------------------|
| Acceptance Ratio | +20.3% |
| E2E Delay | -42.65% |
| Resource Consumption | -50% |

**Baseline**: Rule-based heuristic placing VNFs theo min delay priority