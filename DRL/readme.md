# Deep Reinforcement Learning for SFC Provisioning

Implementation của nghiên cứu "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning" (IEEE Networking Letters, 2024).

## Cấu trúc dự án

```
.
├── config.py                 # Cấu hình từ Tables I, II của nghiên cứu
├── utils.py                  # Network topology, priorities, replay buffer
├── env/
│   └── sfc_environment.py    # Gymnasium environment với ràng buộc C1-C5
├── dqn_model.py             # DQN với attention layer (Section III)
├── agent.py                 # DQN agent với epsilon-greedy
├── main.py                  # Training & testing script
├── baseline_heuristic.py    # Baseline comparison (Figure 2)
├── demo.py                  # Quick demo
├── tests.py                 # Unit tests
└── README.md
```

## Kiến trúc DRL (Section III)

### Input States (3 layers - Algorithm 1):
1. **State 1** `[1 × (2|V| + 2)]`: DC info
   - Installed VNFs per type: n^v_i
   - Available VNFs per type: n^v_i - allocated
   - Available storage & CPU: S_i, Q_i

2. **State 2** `[|S| × (1 + 2|V|)]`: SFC processing stages
   - SFC type indicator
   - Already allocated VNFs
   - Remaining VNFs in chain

3. **State 3** `[|S| × (4 + |V|)]`: Pending SFCs
   - SFC type, count, remaining delay, BW
   - VNF waiting counts

### DNN Architecture (Figure 1):
```
Input1 → FCDNN(128,64)  ┐
Input2 → FCDNN(64,64)   ├→ Concatenate → Attention → Dense(256,128,64) → Actions
Input3 → FCDNN(64,64)   ┘
```

### Output Actions `[2|V| + 1]`:
- **Allocation** (|V| actions): Install + allocate VNF
- **Uninstallation** (|V| actions): Remove idle VNF
- **Wait** (1 action): No operation

## Constraints (Section II - Problem P)

✅ **C1**: CPU constraint - `Σ n^v_i * C_v ≤ Q_i`
✅ **C2**: Storage constraint - `Σ n^v_i * S_v ≤ S_i`
✅ **C3**: VNF uniqueness - Each VNF processed by only one DC
✅ **C4**: Bandwidth constraint - `Σ δ^s_ij * B_s ≤ B_ij`
✅ **C5**: E2E delay - `t^s_Pg + t^s_Pr ≤ D_s`

## Priority System (Algorithm 1, Steps 23-29)

**VNF Priority = P1 + P2 + P3**

- **P1 (RemTime)**: `TE_s - D_s` - Remaining time urgency
- **P2 (SFC_based)**: DC allocation preference
  - +10 if VNFs already in same DC
  - -5 if VNFs in other DCs
- **P3 (Urgency)**: `C / (D_s - TE_s + ε)` if remaining < threshold

## Cài đặt

```bash
pip install gymnasium tensorflow numpy networkx matplotlib
```

## Quick Demo

```bash
python demo.py
```

Chạy 5 episodes nhanh để test environment và agent.

## Debug & Verification

```bash
python debug_test.py
```

Chạy 6 test suites để verify:
1. ✅ Environment basic operations
2. ✅ Agent basic operations  
3. ✅ Training loop
4. ✅ VNF allocation logic
5. ✅ Network reconfigurability
6. ✅ Constraints (C1-C5)

## Full Training

```bash
python main.py
```

**Training Parameters (từ nghiên cứu):**
- Updates (U): 350
- Episodes per update (E): 20  
- Actions per step (A): 100
- Step duration (T): 1 ms
- Request interval (N): 4 steps
- Batch size: 64
- Memory size: 10,000
- Gamma: 0.99
- Epsilon: 1.0 → 0.01 (decay 0.995)

**Reconfigurability:** Mỗi episode train với số DC ngẫu nhiên (2-8) để model học generalize.

## Testing  

```bash
python tests.py -v
```

**Test Coverage:**
- Environment initialization & SFC generation
- Observation/action spaces validation
- VNF allocation logic (Algorithm 1)
- Priority calculations (P1, P2, P3)
- Constraint checking (C1-C5)
- DQN agent behavior
- Replay buffer operations

## Baseline Comparison (Figure 2)

Code tự động so sánh với baseline heuristic:

```python
from baseline_heuristic import compare_with_baseline
compare_with_baseline(agent, num_dcs=4, num_tests=5)
```

**Expected Improvements (từ nghiên cứu):**
- Acceptance Ratio: **+20.3%**
- E2E Delay: **-42.65%**
- Resource Consumption: **-50%**

## Reward System

```python
+2.0  # SFC satisfied within E2E delay
-1.5  # SFC dropped (timeout or constraints)
-1.0  # Invalid action
-0.5  # Uninstall required VNF
 0.0  # Default/Wait
```

## Network Reconfigurability (Key Contribution)

Model hoạt động trên các NC khác nhau **mà không cần retrain**:

1. **Input states độc lập DC count**: States dựa trên VNF types và SFC types, không phụ thuộc |N|
2. **DC iteration động**: DC_priority_list cập nhật mỗi step dựa trên shortest path
3. **Action space cố định**: Actions chỉ phụ thuộc |V|, không phụ thuộc |N|

**Test:** Model train với 4 DCs có thể test ngay trên 2, 6, 8 DCs (Figure 4).

## Results Visualization (Figures 2-4)

Training tự động tạo `training_results.png` với 9 plots:

**Row 1:** Training progress
- Rewards over updates
- Acceptance ratio trend
- Training loss

**Row 2:** Training metrics
- E2E delay evolution
- Resource utilization
- Test acceptance by DC count

**Row 3:** Test results
- Delay vs network size
- Resource util vs network size  
- SFC type acceptance (4 DCs)

## SFC Types (Table I)

| Type | VNF Chain | BW | Delay | Bundle |
|------|-----------|-------|-------|---------|
| **CG** | NAT→FW→VOC→TM | 50 Mbps | 50 ms | 10-20 |
| **AR** | NAT→FW→VOC | 30 Mbps | 20 ms | 15-25 |
| **VS** | NAT→FW→VOC→TM | 40 Mbps | 100 ms | 20-30 |
| **VoIP** | NAT→FW | 10 Mbps | 150 ms | 25-35 |
| **MIoT** | NAT→FW→IDPS | 5 Mbps | 30 ms | 30-40 |
| **Ind4.0** | NAT→FW→IDPS→WO | 20 Mbps | 25 ms | 20-30 |

## VNF Specifications (Table II)

| VNF | CPU | RAM | Storage | Proc Time |
|-----|-----|-----|---------|-----------|
| **NAT** | 2 GHz | 4 GB | 10 GB | 5 ms |
| **FW** | 3 GHz | 6 GB | 15 GB | 8 ms |
| **VOC** | 4 GHz | 8 GB | 20 GB | 12 ms |
| **TM** | 2 GHz | 4 GB | 10 GB | 6 ms |
| **WO** | 3 GHz | 6 GB | 15 GB | 10 ms |
| **IDPS** | 5 GHz | 10 GB | 25 GB | 15 ms |

## DC Configuration

- CPU: 12-120 GHz (random per DC)
- RAM: 256 GB
- Storage: 2 TB
- Link BW: 1 Gbps

## Algorithm Flow (Algorithm 1)

```
1. Set DC priority based on shortest path to min-delay SFC
2. For each DC in priority order:
   3. Get current DC info → State 1
   4. Get DC's SFC processing → State 2
   5. Get overall SFC info → State 3
   6. Model predicts action
   7. If Uninstall:
      - Check if idle VNF exists
      - Check no pending requests need it
   8. If Allocate:
      - Check resource availability
      - Check VNFs waiting for this type
      - Calculate priorities (P1+P2+P3)
      - Select highest priority VNF
      - Perform allocation
   9. Calculate rewards, update states
   10. Store to replay memory
```

## Key Implementation Details

1. **Sequential VNF Execution**: VNFs allocated in exact chain order
2. **Waiting Time**: Calculated based on queue length at each DC
3. **Propagation Delay**: Distance / speed of light (fiber optic)
4. **Processing Delay**: Sum of all VNF processing times + waiting
5. **BW Management**: Links tracked for available bandwidth
6. **Resource Release**: Automatic when SFC satisfied or dropped

## Verification Checklist

- [x] 3-layer input architecture (Section III)
- [x] Attention layer in DNN
- [x] All 5 constraints (C1-C5) enforced
- [x] Priority system (P1, P2, P3) implemented
- [x] Algorithm 1 flow complete
- [x] DC priority ordering
- [x] Network reconfigurability
- [x] Baseline comparison
- [x] Visualization matching Figures 2-4
- [x] All parameters from paper

## Citation

```bibtex
@article{onsu2024unlocking,
  title={Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning},
  author={Onsu, Murat Arda and Lohan, Poonam and Kantarci, Burak and Janulewicz, Emil and Slobodrian, Sergio},
  journal={IEEE Networking Letters},
  volume={6},
  number={3},
  pages={193--197},
  year={2024},
  publisher={IEEE}
}
```

## License

Research implementation for academic purposes.