# Training Flow - Chi tiết theo Paper

## Paper Parameters (Section III.A.4)

```python
U = 10   # Total Updates (config.TRAIN_UPDATES)
E = 20   # Episodes per Update (config.EPISODES_PER_UPDATE)
A = 100  # Actions per timestep (config.ACTIONS_PER_TIME_STEP)
T = 1ms  # Timestep duration (config.TIME_STEP)
```

## ⏱️ Timing Model (CRITICAL)

### Paper Quote:
> "The model performs A actions (i.e., A = 100), during each step, which is of duration T, 1 ms. Every action inference time stamp is 0.01ms."

### Visual Explanation:

```
┌────────────────────────────────────────────────────────────────────┐
│                   1 TIMESTEP (T = 1 millisecond)                   │
├────────┬────────┬────────┬────────┬─────┬────────┬────────┬────────┤
│ Act 1  │ Act 2  │ Act 3  │ Act 4  │ ... │ Act 98 │ Act 99 │ Act 100│
│0.01ms  │0.01ms  │0.01ms  │0.01ms  │     │0.01ms  │0.01ms  │0.01ms  │
└────────┴────────┴────────┴────────┴─────┴────────┴────────┴────────┘
         ↑ Model inference time per action = T/A = 1ms/100 = 0.01ms

After 100 actions complete → Physical time advances by 1ms:
├─ VNF processing times decrement by 1ms
├─ Request elapsed times increment by 1ms
├─ Check for E2E delay violations
└─ Generate new traffic (if at interval)
```

### Key Points:

1. **Action Time vs Physical Time**:
   - Action inference: 0.01ms (logical, for model decision)
   - Physical time: Only advances after A=100 actions
   - This allows model to make multiple decisions within 1ms of simulation time

2. **Why This Design?**:
   ```
   Real-world scenario:
   - DRL model makes decisions very fast (~0.01ms)
   - But VNF processing takes longer (0.06-0.11ms)
   - So model can make 100 placement decisions while 1 VNF is processing
   ```

3. **Implementation**:
   ```python
   action_counter = 0
   
   for action in range(100):  # A = 100 actions
       # Model inference (0.01ms each)
       state = get_state()
       action = model.predict(state)
       execute_action(action)
       action_counter += 1
   
   # After 100 actions:
   if action_counter >= 100:
       sim_time += 1  # Physical time advances by T=1ms
       update_vnf_processing()  # Decrement by 1ms
       update_request_times()    # Increment by 1ms
       action_counter = 0
   ```

### Code Configuration:

```python
# Paper values:
ACTIONS_PER_TIME_STEP = 100  # A (paper)
TIME_STEP = 1                # T = 1ms
# → Action inference time = 1ms / 100 = 0.01ms

# Current code (faster training):
ACTIONS_PER_TIME_STEP = 50   # Modified for efficiency
TIME_STEP = 1                # T = 1ms
# → Action inference time = 1ms / 50 = 0.02ms
```

### Example Timeline:

```
Episode starts (sim_time = 0ms):
├─ Traffic generated: 10 SFC requests
└─ action_counter = 0

Action 1 (sim_time still 0ms):
├─ DC 0: Allocate NAT for Request #1
├─ action_counter = 1
└─ Physical time: UNCHANGED

Action 2 (sim_time still 0ms):
├─ DC 1: Allocate FW for Request #2
├─ action_counter = 2
└─ Physical time: UNCHANGED

...

Action 50 (sim_time still 0ms):
├─ DC 2: Allocate TM for Request #5
├─ action_counter = 50
└─ ⚠️ TIMESTEP COMPLETE! Physical time advances:
    ├─ sim_time = 0 → 1ms
    ├─ All VNF processing times decrement by 1ms
    │  Example: NAT (proc_time: 0.06ms → 0ms, now idle)
    ├─ All request elapsed_times increment by 1ms
    ├─ Check for drops (elapsed_time > max_delay)
    └─ action_counter = 0 (reset)

Action 51 (sim_time = 1ms):
├─ DC 3: Allocate VOC for Request #1
└─ action_counter = 1

...

Action 100 (sim_time = 1ms):
├─ action_counter = 50
└─ TIMESTEP COMPLETE!
    ├─ sim_time = 1 → 2ms
    └─ ...

sim_time = 4ms:
└─ TRAFFIC GENERATION INTERVAL (every N=4ms)
    ├─ Generate new batch of requests
    └─ Continue...

Episode ends when:
├─ sim_time > MAX_SIM_TIME (100ms) OR
└─ (sim_time > TRAFFIC_STOP_TIME (150ms) AND no active requests)
```

## Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PROCESS                         │
│                  (10 Updates x 20 Episodes = 200 Episodes)  │
└─────────────────────────────────────────────────────────────┘

UPDATE 1 (Epsilon = 1.0)
├── Episode 1  → Collect experience → Add to replay memory
├── Episode 2  → Collect experience → Add to replay memory
├── ...
├── Episode 20 → Collect experience → Add to replay memory
└── [TRAIN MODEL] ← Sample batch from replay memory & update weights
    └── Target network: No update (update every 5 updates)

UPDATE 2 (Epsilon = 0.9)
├── Episode 21 → Collect experience → Add to replay memory
├── ...
├── Episode 40 → Collect experience → Add to replay memory
└── [TRAIN MODEL] ← Update weights

UPDATE 3-4 (Epsilon decreasing...)
├── ...

UPDATE 5 (Epsilon = 0.66)
├── Episode 81-100
└── [TRAIN MODEL] + [UPDATE TARGET NETWORK] ← Every 5 updates

UPDATE 6-9
├── ...

UPDATE 10 (Epsilon = 0.35)
├── Episode 181-200
└── [TRAIN MODEL] + [UPDATE TARGET NETWORK]

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING COMPLETED                        │
│    Total: 200 episodes, 10 weight updates, 2 target updates │
└─────────────────────────────────────────────────────────────┘
```

## Timeline Chi tiết

### UPDATE 1 (Episodes 1-20)

**Epsilon = 1.0** (100% exploration)

```
Episode 1:
  ├── Reset env → Random DC count (2-6)
  ├── Generate initial SFC requests
  ├── Loop until done:
  │   ├── Get state + action mask
  │   ├── Choose action (random with mask)
  │   ├── Execute action (100 actions per 1ms)
  │   ├── Receive reward
  │   └── Store (state, action, reward, next_state, done)
  └── Episode done → Store all transitions to replay memory

Episode 2-20: (tương tự)

After Episode 20:
  └── TRAIN MODEL:
      ├── Sample batch_size=64 from replay memory
      ├── Calculate Q-targets using target network
      ├── Update main network weights
      └── Epsilon decay: 1.0 → 0.9
```

### UPDATE 2 (Episodes 21-40)

**Epsilon = 0.9** (90% exploration, 10% exploitation)

```
Episode 21:
  ├── 90% actions: Random (from valid actions)
  ├── 10% actions: argmax Q-values (from valid actions)
  └── Model starts learning patterns

...

After Episode 40:
  └── TRAIN MODEL → Epsilon: 0.9 → 0.81
```

### UPDATE 5 (Episodes 81-100)

**Epsilon = 0.66** (66% exploration, 34% exploitation)

```
After Episode 100:
  ├── TRAIN MODEL
  └── UPDATE TARGET NETWORK ← Important: Stabilizes training
      └── target_model.weights ← main_model.weights
```

### UPDATE 10 (Episodes 181-200)

**Epsilon = 0.35** (35% exploration, 65% exploitation)

```
Episode 200:
  └── Model mostly exploits learned policy

After Episode 200:
  ├── TRAIN MODEL (final time)
  ├── UPDATE TARGET NETWORK
  └── Save final weights
```

## Key Concepts

### 1. Episode vs Update

- **Episode**: Một lần chạy hoàn chỉnh từ reset() đến done
  - Mỗi episode có nhiều timesteps
  - Mỗi timestep có 100 actions
  - Kết thúc khi: time > MAX_TIME hoặc (time > STOP_TIME và no requests)

- **Update**: Một lần training model (update weights)
  - Sau E=20 episodes → Train 1 lần
  - Total U=10 updates trong toàn bộ training

### 2. Replay Memory

```python
replay_memory = deque(maxlen=10000)

# Trong mỗi episode:
transitions = []  # (state, action, reward, next_state, done)
# ... collect transitions ...
replay_memory.extend(transitions)

# Sau E episodes → Train:
batch = random.sample(replay_memory, 64)
# ... update weights using batch ...
```

### 3. Epsilon Decay

```python
epsilon = 1.0  # Start
epsilon_decay = 0.9
epsilon_min = 0.1

# Sau MỖI UPDATE (không phải mỗi episode):
if epsilon > epsilon_min:
    epsilon *= epsilon_decay

# Timeline:
# Update 1: ε = 1.0
# Update 2: ε = 0.9
# Update 3: ε = 0.81
# Update 4: ε = 0.73
# Update 5: ε = 0.66
# Update 6: ε = 0.59
# Update 7: ε = 0.53
# Update 8: ε = 0.48
# Update 9: ε = 0.43
# Update 10: ε = 0.39
```

### 4. Target Network

```python
# Main network: Updated mỗi update
main_model.train(batch)

# Target network: Updated mỗi 5 updates
if update_cnt % 5 == 0:
    target_model.set_weights(main_model.get_weights())

# Tại sao cần target network?
# - Prevent moving target problem
# - Stabilize Q-learning
# - Giảm oscillation trong training
```

## Output Example

```
=== Starting SFC Provisioning Training ===
Total Updates: 10
Episodes per Update: 20
Total Episodes: 200

Training Flow:
  - Run 20 episodes consecutively
  - After 20 episodes → Train model (1 update)
  - Repeat 10 times
================================================================================

   [Update 1/10] Episode 1/20: Reward=-234.5, AR=45.23%, DCs=4, Epsilon=1.000
   [Update 1/10] Episode 2/20: Reward=-189.2, AR=52.10%, DCs=3, Epsilon=1.000
   ...
   [Update 1/10] Episode 20/20: Reward=-102.3, AR=68.45%, DCs=5, Epsilon=1.000

[Update 1/10 COMPLETED] Avg Reward: -165.32 | Avg AR: 58.23% | Loss: 2.3421 | Epsilon: 0.9000
--------------------------------------------------------------------------------

   [Update 2/10] Episode 1/20: Reward=-145.6, AR=62.11%, DCs=4, Epsilon=0.900
   ...

[Update 5/10 COMPLETED] Avg Reward: -89.45 | Avg AR: 75.12% | Loss: 1.2341 | Epsilon: 0.6561
   >>> Target network updated <<<
   [Trend] Last 50 eps: AR=71.23%, Reward=-112.34
--------------------------------------------------------------------------------

...

[Update 10/10 COMPLETED] Avg Reward: -45.23 | Avg AR: 84.56% | Loss: 0.5234 | Epsilon: 0.3874
   >>> Best AR so far: 86.12% - saved! <<<
   >>> Target network updated <<<
   [Trend] Last 50 eps: AR=82.45%, Reward=-62.11
--------------------------------------------------------------------------------

Training progress plot saved to: training_progress.png

================================================================================
Training Completed!
Total Episodes Run: 200
Overall Avg AR: 72.34%
Overall Avg Reward: -98.56
Final model saved to: sfc_dqn_weights.weights.h5
Best model saved to: best_sfc_dqn_weights.weights.h5 (AR=86.12%)
================================================================================
```

## Common Mistakes (FIXED)

### ❌ Sai: Train sau mỗi episode
```python
for episode in range(200):
    # ... run episode ...
    agent.train()  # WRONG!
```

### ✅ Đúng: Train sau E episodes
```python
for update in range(10):
    for episode in range(20):
        # ... run episode ...
    agent.train()  # CORRECT!
```

### ❌ Sai: Decay epsilon mỗi episode
```python
for episode in range(200):
    # ... run episode ...
    epsilon *= decay  # WRONG!
```

### ✅ Đúng: Decay epsilon mỗi update
```python
for update in range(10):
    for episode in range(20):
        # ... run episode ...
    agent.train()
    epsilon *= decay  # CORRECT!
```

## Monitoring Training

### Indicators of Good Training

1. **Reward tăng dần**: -200 → -100 → -50 → 0 → +100
2. **AR tăng dần**: 40% → 60% → 70% → 80%+
3. **Loss giảm dần**: 5.0 → 2.0 → 1.0 → 0.5
4. **Epsilon giảm đều**: 1.0 → 0.9 → ... → 0.1

### Red Flags

1. **Reward không tăng** sau 5 updates → Check State 2
2. **AR = 0%** liên tục → Check action masking
3. **Loss tăng** hoặc NaN → Giảm learning rate
4. **Model chọn toàn WAIT** → Check reward logic