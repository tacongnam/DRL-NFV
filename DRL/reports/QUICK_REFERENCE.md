# Quick Reference - SFC Provisioning DRL

## ⏱️ Timing Architecture (CRITICAL!)

### Three Time Scales:

| Scale | Value | What Changes |
|-------|-------|--------------|
| **Action Inference** | 0.01ms (or 0.02ms*) | Model makes decision |
| **Physical Timestep** | 1ms | VNF processing, request timers |
| **Episode** | 100-200ms | Complete episode duration |

*Note: Paper uses A=100 (0.01ms), code uses A=50 (0.02ms) for faster training

### Key Rule:
```
Physical time advances ONLY after A actions complete!

Actions 1-50:  sim_time = 0ms (unchanged)
Action 50:     ← Timestep boundary!
               sim_time: 0ms → 1ms
               VNF processing times: -1ms
               Request elapsed times: +1ms
Actions 51-100: sim_time = 1ms (unchanged)
Action 100:    ← Next timestep!
               sim_time: 1ms → 2ms
```

### Timeline Example:
```
Action Count:  1  2  3  ...  50 | 51 52 ... 100 | 101 ...
               ├──────────────┤   ├─────────────┤
Physical Time: |   0ms        | → |    1ms      | → 2ms
                              ↑                   ↑
                           Timestep           Timestep
                           advances           advances
```

## 📊 Training Parameters (From Paper)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| U (Updates) | 10 | Số lần update weights |
| E (Episodes/Update) | 20 | Số episodes trước mỗi update |
| **Total Episodes** | **200** | **U × E = 10 × 20** |
| A (Actions/Timestep) | 100 | Số actions mỗi 1ms |
| T (Timestep) | 1 ms | Độ dài mỗi timestep |
| Batch Size | 64 | Mini-batch size cho training |
| Memory Size | 10,000 | Replay memory capacity |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon Decay | 0.9 | Decay rate **per update** |
| Epsilon Min | 0.1 | Minimum exploration rate |
| Learning Rate | 0.001 | Adam optimizer LR |
| Gamma (γ) | 0.99 | Discount factor |

## 🔄 Training Flow

```
FOR update = 1 to 10:
    FOR episode = 1 to 20:
        ├── Run episode (collect transitions)
        └── Add to replay memory
    └── Train model (1 weight update)
    └── Decay epsilon
    └── If update % 5 == 0: Update target network
```

## 🎯 Action Space (13 actions)

| Action ID | Type | Description |
|-----------|------|-------------|
| 0 | WAIT | Do nothing this timestep |
| 1-6 | UNINSTALL | Uninstall VNF type 1-6 (NAT, FW, VOC, TM, WO, IDPS) |
| 7-12 | ALLOC | Allocate VNF type 1-6 to highest priority request |

## 🏆 Reward Structure

| Event | Reward | Notes |
|-------|--------|-------|
| SFC Satisfied | +2.0 | All VNFs allocated, within delay |
| SFC Dropped | -1.5 | Exceeded E2E delay |
| Invalid Action | -1.0 | Only for critical errors (no resource) |
| Uninstall Required VNF | -0.5 | Removing needed VNF |
| Wait (no requests) | 0.0 | **FIXED**: Was penalty, now neutral |
| Alloc (no demand) | 0.0 | **FIXED**: Exploration, neutral |
| Uninstall (no VNF) | 0.0 | **FIXED**: Minor issue, neutral |
| Partial Progress | +0.5 | VNF allocated but SFC not done |

## 🎭 State Representation

### State 1: DC Info `[1 × 14]`
```
[cpu, storage, installed_NAT, ..., installed_IDPS, idle_NAT, ..., idle_IDPS]
```

### State 2: DC-SFC Info `[6 × 13] → [78]` ✅ FIXED
```
For each SFC type:
[count, allocated_VNF1, ..., allocated_VNF6, remaining_VNF1, ..., remaining_VNF6]
```

### State 3: Global Info `[6 × 10] → [60]`
```
For each SFC type:
[req_count, avg_remaining_time, bw_req, pending_vnfs, vnf1_count, ..., vnf6_count]
```

## 🛡️ Action Masking (New!)

```python
valid_mask[action] = {
    True:  Action is valid and safe to execute
    False: Action should not be chosen
}

# Prevents:
# - Allocating when no demand
# - Uninstalling urgent VNFs
# - Actions without resources
```

## 📈 Expected Training Progression

| Update | Episodes | Epsilon | Avg Reward | Avg AR | Notes |
|--------|----------|---------|------------|--------|-------|
| 1 | 1-20 | 1.0 → 0.9 | -200 | 50% | Random exploration |
| 2-3 | 21-60 | 0.9 → 0.73 | -150 | 60% | Learning patterns |
| 4-6 | 61-120 | 0.73 → 0.53 | -80 | 70% | Exploitation starts |
| 7-9 | 121-180 | 0.53 → 0.39 | -30 | 75% | Mostly exploiting |
| 10 | 181-200 | 0.39 → 0.35 | +20 | 80%+ | Near-optimal |

## 🚀 Running the Code

### Training
```bash
python train.py

# Expected output:
# - 200 episodes total
# - Reward improving
# - AR increasing
# - Files: sfc_dqn_weights.weights.h5, best_*.h5, training_progress.png
```

### Testing
```bash
python test.py

# Expected output:
# - Exp 1: AR per SFC type (4 DCs)
# - Exp 2: AR vs DC count (2,4,6,8 DCs)
# - Files: result_exp1_fig2.png, result_exp2_fig3.png
```

## 🐛 Debugging Checklist

- [ ] **AR = 0% in test?**
  - Load `best_*.h5` instead of final model
  - Check State 2 is not zeros
  - Enable DEBUG_MODE in test.py

- [ ] **Reward too negative?**
  - Check action masking is working
  - Verify neutral rewards for exploration
  - Should see >80% valid actions

- [ ] **Loss = NaN?**
  - Reduce learning rate
  - Check for overflow in Q-values
  - Verify batch sampling

- [ ] **Model always WAITs?**
  - Check reward structure
  - Verify traffic generation
  - Enable action masking

## 📝 Key Files

```
project/
├── config.py           # All hyperparameters (DON'T MODIFY)
├── train.py           # Training loop (10 updates × 20 episodes)
├── test.py            # Testing with 2 experiments
├── env/
│   ├── network.py     # Environment (State 2 FIXED, Action Masking NEW)
│   ├── dqn.py         # DQN model (Action Masking NEW)
│   ├── sfc.py         # SFC Manager
│   └── utils.py       # Helper classes (DC, VNF, Priority)
└── outputs/
    ├── sfc_dqn_weights.weights.h5      # Final model
    ├── best_sfc_dqn_weights.weights.h5 # Best model
    ├── training_progress.png            # Training curves
    ├── result_exp1_fig2.png            # Test results
    └── result_exp2_fig3.png            # Reconfigurability test
```

## 💡 Pro Tips

1. **Always use best_*.h5 for testing** - Final model might overfit
2. **Monitor loss AND AR** - Both should improve
3. **Check action distribution** - Should see diverse actions, not just WAIT
4. **State 2 is critical** - Model blind without it
5. **Action masking saves training time** - Prevents wasted exploration
6. **Target network updates matter** - Don't skip this!

## 🎓 Paper References

- Section III.A: DRL Architecture → State/Action definition
- Section III.A.4: Training → U, E, A, T parameters
- Section III.B: Algorithm 1 → Priority calculation (P1, P2, P3)
- Section IV: Performance → Expected metrics

---

**Last Updated**: With State 2 fix, Action Masking, and Reward refinement