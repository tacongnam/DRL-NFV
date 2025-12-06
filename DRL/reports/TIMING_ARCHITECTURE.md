# Timing Architecture - SFC Provisioning DRL

## Paper Specification (Section III.A.4)

> "The model performs **A actions** (i.e., A = 100), during each step, which is of duration **T, 1 ms**. Every action inference time stamp is **0.01ms**."

## Three Time Scales

### 1. Action Inference Time (0.01ms per action)
- **Model decision time**: How fast DRL model can make a decision
- Formula: `T / A = 1ms / 100 = 0.01ms`
- This is the **logical time** for model inference

### 2. Physical Timestep (1ms per timestep)
- **Simulation time unit**: Smallest unit of physical time
- After A=100 actions complete → Physical time advances by T=1ms
- VNF processing, request timers operate on this scale

### 3. Episode Duration (100-200ms typical)
- Complete lifecycle of one training episode
- From env.reset() to done=True
- Includes many timesteps

## Complete Timeline Example

```
═══════════════════════════════════════════════════════════════════════
EPISODE 1 START (Training Update 1, Episode 1/20)
═══════════════════════════════════════════════════════════════════════

┌─ sim_time = 0ms ─────────────────────────────────────────────────┐
│ Initial State:                                                    │
│ ├─ Generate traffic: 15 SFC requests                             │
│ ├─ 4 DCs available                                               │
│ └─ action_counter = 0                                            │
└───────────────────────────────────────────────────────────────────┘

┌─ TIMESTEP 1: sim_time = 0ms → 1ms ──────────────────────────────┐
│                                                                   │
│ Action 1 (inference: 0.01ms):                                    │
│   ├─ State: DC0, pending_reqs=15                                 │
│   ├─ Model chooses: ALLOC-NAT (action=7)                         │
│   ├─ Execute: Allocate NAT to Request #3 at DC0                  │
│   │   └─ NAT instance created, proc_time = 0.06ms                │
│   ├─ Reward: +0.5 (partial progress)                             │
│   └─ action_counter = 1                                          │
│                                                                   │
│ Action 2 (inference: 0.01ms):                                    │
│   ├─ Current DC: DC1                                             │
│   ├─ Model chooses: ALLOC-FW (action=8)                          │
│   └─ action_counter = 2                                          │
│                                                                   │
│ ... (Actions 3-50) ...                                           │
│                                                                   │
│ Action 50 (inference: 0.01ms):                                   │
│   ├─ Model chooses: ALLOC-TM                                     │
│   ├─ action_counter = 50                                         │
│   └─ ⚠️ TIMESTEP COMPLETE! ⚠️                                     │
│       │                                                           │
│       ├─ Physical time advances: 0ms → 1ms                       │
│       │                                                           │
│       ├─ Update VNF processing times:                            │
│       │   ├─ NAT at DC0: 0.06ms → 0ms (DONE, now idle)          │
│       │   ├─ FW at DC1: 0.03ms → 0ms (DONE, now idle)           │
│       │   └─ TM at DC2: 0.07ms → 0ms (still processing)         │
│       │                                                           │
│       ├─ Update request times:                                   │
│       │   ├─ Request #3: elapsed_time = 0ms → 1ms               │
│       │   │   Check: 1ms ≤ 80ms (max_delay) ✓ OK                │
│       │   └─ ... (all other requests)                            │
│       │                                                           │
│       ├─ Clean completed/dropped requests:                       │
│       │   └─ Request #3 chain: NAT✓ FW→ VOC→ WO→ IDPS           │
│       │       (not complete yet, continue)                       │
│       │                                                           │
│       └─ Traffic generation:                                     │
│           └─ sim_time=1ms % 4 ≠ 0, no new traffic               │
│                                                                   │
│       └─ action_counter = 0 (reset)                             │
└───────────────────────────────────────────────────────────────────┘

┌─ TIMESTEP 2: sim_time = 1ms → 2ms ──────────────────────────────┐
│                                                                   │
│ Action 51-100: (Same pattern)                                    │
│   └─ Physical time: 1ms → 2ms                                    │
└───────────────────────────────────────────────────────────────────┘

┌─ TIMESTEP 3: sim_time = 2ms → 3ms ──────────────────────────────┐
│ Action 101-150                                                    │
└───────────────────────────────────────────────────────────────────┘

┌─ TIMESTEP 4: sim_time = 3ms → 4ms ──────────────────────────────┐
│ Action 151-200                                                    │
│   └─ Physical time: 3ms → 4ms                                    │
│       └─ Traffic generation triggered!                           │
│           (sim_time=4ms % 4 == 0)                                │
│           └─ Generate new batch: 12 requests                     │
│               Total pending: 15 - 3 (completed) + 12 = 24        │
└───────────────────────────────────────────────────────────────────┘

... Continue until episode done ...

┌─ TIMESTEP 100: sim_time = 99ms → 100ms ─────────────────────────┐
│ Action 4951-5000                                                  │
│   └─ sim_time reaches MAX_SIM_TIME_PER_EPISODE                  │
│       └─ Episode terminates (done = True)                        │
└───────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
EPISODE 1 END
═══════════════════════════════════════════════════════════════════════
Final Stats:
├─ Total actions taken: ~5000
├─ Total timesteps: 100
├─ Episode duration: 100ms (simulation time)
├─ Requests generated: 150
├─ Requests completed: 120 (AR = 80%)
├─ Requests dropped: 30
└─ Total reward: +85.5

Store all transitions to replay memory...
Continue to Episode 2...
```

## Key Observations

### 1. Decoupling of Time Scales

```python
# Action time (logical):
for i in range(100):  # 100 actions
    action = model.predict(state)  # 0.01ms each (logical)
    execute(action)                # Immediate state change
    # BUT physical time has NOT advanced yet!

# Physical time (after 100 actions):
sim_time += 1  # Now advance by 1ms
vnf.processing_time -= 1  # VNF processing decrements
request.elapsed_time += 1  # Request timer increments
```

### 2. Why This Design?

**Real-world analogy:**
```
Human decision making:
├─ Think about 100 different strategies (fast, in your head)
├─ Choose the best one
└─ Execute it (takes actual time)

DRL model:
├─ Evaluate 100 possible VNF placements (0.01ms each = 1ms total)
├─ Choose best placement
└─ VNF processes (actual 0.06-0.11ms)
```

**Benefits:**
- Model can make multiple strategic decisions per physical timestep
- More responsive to changing network conditions
- Better exploration of action space
- Matches reality: Control decisions are faster than execution

### 3. VNF Processing vs Action Time

```
VNF Processing Times (from Paper Table II):
├─ NAT:  0.06ms
├─ FW:   0.03ms
├─ VOC:  0.11ms
├─ TM:   0.07ms
├─ WO:   0.08ms
└─ IDPS: 0.02ms

Action Inference Time:
└─ 0.01ms per action (100 actions per 1ms timestep)

Observation:
- All VNF processing times < 1ms (one timestep)
- Most complete within one timestep
- VOC is slowest (0.11ms) but still < 1ms
```

### 4. Traffic Generation Pattern

```
Time:  0ms   4ms   8ms   12ms  16ms  ...  150ms  154ms  ...
       │     │     │     │     │           │      │
Gen:   ●─────●─────●─────●─────●───────────●──────┴─(stop)
       │     │     │     │     │           │
       10req 15req 12req 18req 14req  ...  8req   (no more)
       
Episode continues until:
├─ sim_time > 150ms AND
└─ All active requests processed (completed or dropped)
```

## Code Validation

### Checking Timing Consistency:

```python
# In your environment:
def validate_timing(self):
    """Check if timing parameters are consistent"""
    
    action_time = config.TIME_STEP / config.ACTIONS_PER_TIME_STEP
    print(f"Action inference time: {action_time:.4f}ms")
    
    # Check if any VNF proc_time is shorter than action_time
    for vnf, specs in config.VNF_SPECS.items():
        proc_time = specs['proc_time']
        if proc_time < action_time:
            print(f"⚠️  {vnf} proc_time ({proc_time}ms) < action_time ({action_time}ms)")
        
        # Check how many timesteps needed
        timesteps_needed = proc_time / config.TIME_STEP
        print(f"   {vnf}: {proc_time}ms = {timesteps_needed:.2f} timesteps")
```

### Expected Output:
```
Action inference time: 0.0200ms  (with A=50 in config)
   NAT: 0.06ms = 0.06 timesteps  (completes in 1 timestep)
   FW: 0.03ms = 0.03 timesteps
   VOC: 0.11ms = 0.11 timesteps
   TM: 0.07ms = 0.07 timesteps
   WO: 0.08ms = 0.08 timesteps
   IDPS: 0.02ms = 0.02 timesteps
```

## Debugging Tips

### 1. Enable Timing Debug:
```python
# In test.py:
DEBUG_TIMING = True

# Output will show:
# Step 10: Action=ALLOC-NAT, SimTime=0ms, ActionInTimestep=11/50, Pending=15
# Step 20: Action=ALLOC-FW, SimTime=0ms, ActionInTimestep=21/50, Pending=14
# Step 50: Action=ALLOC-TM, SimTime=0ms, ActionInTimestep=50/50, Pending=12
#   → [Timestep advances: 0ms → 1ms]
# Step 51: Action=WAIT, SimTime=1ms, ActionInTimestep=1/50, Pending=12
```

### 2. Track Timestep Boundaries:
```python
if self.action_step_counter == config.ACTIONS_PER_TIME_STEP - 1:
    print(f"⚠️  Next action will trigger timestep advance!")
    print(f"    Current: {self.sim_time_ms}ms → Next: {self.sim_time_ms + 1}ms")
```

### 3. Monitor VNF Processing:
```python
for vnf in dc.installed_vnfs:
    if vnf.remaining_proc_time > 0:
        print(f"VNF {vnf.vnf_type} at DC{vnf.dc_id}: "
              f"{vnf.remaining_proc_time:.2f}ms remaining")
```

## Summary

| Concept | Value | Formula | Meaning |
|---------|-------|---------|---------|
| **Action Inference Time** | 0.01ms | T/A = 1ms/100 | Time for model to choose one action |
| **Physical Timestep** | 1ms | T = 1ms | Time for VNF processing, request aging |
| **Actions per Timestep** | 100 | A = 100 | How many decisions before time advances |
| **Traffic Interval** | 4ms | N = 4 | Generate new requests every 4 timesteps |
| **Episode Duration** | 100-200ms | Varies | Total simulation time per episode |

**Critical Rule**: Physical time (`sim_time_ms`) only advances after `ACTIONS_PER_TIME_STEP` actions complete!