# GenAI-DRL SFC Provisioning

## Algorithm Flow

### Input Format
- **V**: Nodes (DataCenter + SwitchNode) with resources (cpu, ram, storage, delay)
- **E**: Links (u, v, bandwidth, delay)
- **F**: VNF types (cpu, ram, storage, startup_time per DC)
- **R**: SFC requests (arrival_time, source, destination, vnf_chain, bandwidth, max_delay)

**JSON Structure**:
```json
{
  "V": {
    "0": {"server": true, "c_v": 30, "r_v": 40, "h_v": 35, "d_v": 0.2},
    "1": {"server": false}
  },
  "E": [
    {"u": 0, "v": 1, "b_l": 80, "d_l": 0.05}
  ],
  "F": [
    {"c_f": 1.2, "r_f": 1.0, "h_f": 0.8, "d_f": {"0": 0.3, "1": 0.4}}
  ],
  "R": [
    {"T": 0, "st_r": 2, "d_r": 4, "F_r": [0, 1], "b_r": 1.5, "d_max": 2.0, "type": "Optional"}
  ]
}
```

**Notes**:
- `F_r`: VNF chain as list of indices [0, 1, ...] (not SFC type names)
- `type`: Optional field for post-analysis only (not used in training)
- Typical dataset: 10 VNF types, 2500+ unique chains, mostly 1-2 VNFs per chain

### Preprocessing
- Build NetworkX graph G(V, E)
- Precompute all-pairs shortest paths between DataCenters (ignoring SwitchNodes as endpoints)
- Cache delay and min_bandwidth for each DC pair

### DRL Training (Priority-based DC Selection)
1. Load requests sorted by arrival_time
2. For each timestep:
   - Activate requests where arrival_time ≤ current_time
   - Calculate DC priority: P1 (urgency) + P2 (locality) + P3 (resource)
   - Sort DCs by priority
   - For each DC (priority order):
     - Observe state: (DC_info, DC_VNF_demand, Global_VNF_demand)
     - DQN selects action: WAIT | UNINSTALL(vnf) | ALLOCATE(vnf)
     - Execute action → reward
   - Advance time → check timeouts
3. Update DQN every N steps, target network every M steps

### GenAI Data Collection
1. Load pre-trained DQN weights
2. Run simulation with **random DC selection** (not priority)
3. For each step:
   - Record: (DC_prev_state, DC_next_state, value)
   - Value = f(urgency, source_count, resource_availability)
4. Store transitions → VAE dataset

### GenAI Training
1. **VAE Training**: Learn DC_prev_state → DC_next_state mapping
   - Encoder: state → latent z (mean, log_var)
   - Decoder: z → reconstructed next_state
   - Loss: MSE(reconstruction) + KL_divergence
2. **Value Network Training**: Learn z → importance_value
   - Input: latent z from frozen encoder
   - Target: heuristic value from data collection
   - Loss: MSE(predicted_value, target_value)
3. Normalize values (mean, std) for stable inference

### GenAI-DRL Inference
1. For each timestep:
   - Get all DC current states
   - Encode → latent z
   - Predict values via Value Network
   - Sort DCs by predicted value (descending)
   - For each DC (VAE order):
     - DQN observes and acts
2. Same reward/update logic as DRL

## Routing
- VNF placement on DC creates propagation delay via shortest path
- Bandwidth allocated on path edges
- Reward penalty: α·delay + β·hop_count

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STEP 1: Train DRL                        │
│  Input: JSON (V,E,F,R) → DRLEnv (Priority DC Selection)    │
│  Output: models/best_sfc_dqn.weights.h5                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: Collect Data for GenAI                 │
│  Load: DRL weights → VAEEnv (Random DC Selection)          │
│  Collect: (DC_prev, DC_next, value) × N samples            │
│  Output: GenAI dataset in memory                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           STEP 2.1: Train VAE (state prediction)            │
│  Loss: MSE(DC_next, decoder(encoder(DC_prev))) + KL        │
│  Output: Encoder weights (frozen for Value Net)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│      STEP 2.2: Train Value Network (importance score)       │
│  Input: latent z = encoder(DC_state)                        │
│  Loss: MSE(predicted_value, heuristic_value)               │
│  Output: models/genai_model_{encoder,decoder,value}.h5     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: Train GenAI-DRL                        │
│  Load: GenAI model → VAEEnv (VAE DC Selection)             │
│  DC Selection: argmax(ValueNet(encoder(DC_states)))        │
│  Output: models/best_genai_sfc_dqn.weights.h5              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    STEP 4: Evaluate                         │
│  Compare: DRL vs GenAI-DRL on SFC AccRatio, E2E, Throughput│
│  Output: fig/{genai_}result_exp1.png, exp2.png             │
└─────────────────────────────────────────────────────────────┘
```

## Commands
```bash
# Step 1: Train DRL with priority-based selection
python scripts.py train

# Step 2: Collect data for GenAI (random selection)
python scripts.py collect

# Step 3: Train GenAI-DRL
python scripts.py train --mode genai

# Step 4: Evaluate
python scripts.py eval                # DRL
python scripts.py eval --mode genai   # GenAI-DRL
```

## Key Differences from Paper Implementation

### Routing
- Paper assumes full DC mesh with direct links
- Implementation uses NetworkX graph with SwitchNodes
- Shortest path computed via Dijkstra with bandwidth constraints
- Path metrics cached at initialization (Floyd-Warshall on DCs)

### DC Selection
- **DRL (Priority)**: Selects DC based on P1(urgency) + P2(locality) + P3(resource)
- **GenAI-DRL**: VAE predicts next_state, Value Network ranks DCs by importance

### Request Arrival
- Paper uses bulk generation every N ms
- Implementation uses individual arrival_time per request (sorted queue)

## Key Files (Updated for Dynamic VNF)
- **config.py**: `update_vnf_specs()` loads from JSON, dynamic ACTION_SPACE_SIZE
- **core/**: dc.py (Node, DataCenter, SwitchNode), vnf.py, request.py, sfc_manager.py, topology.py (cached routing), routing.py, simulator.py
- **envs/observer.py**: `get_state_dim()` = 3 + 2×NUM_VNF_TYPES + 3, aggregated VNF demand
- **agents/dqn_model.py**: Dynamic input/output shapes based on NUM_VNF_TYPES
- **envs/**: base_env.py, drl_env.py (priority), vae_env.py (random/GenAI), controller.py, priority.py, vae_selector.py, vae_trainer.py, utils.py
- **runners/experiments.py**: `run_experiment_overall()` - no per-SFC-type breakdown
- **runners/**: train_drl.py, collect_data.py, train_vae.py, eval_drl.py, eval_vae.py, core.py, visualization.py

## Optimizations Applied
- Precompute global stats O(N_req) instead of per-DC O(N_dc × N_req)
- Vectorized VAE inference (all DCs in one call)
- Circular buffer for VAE dataset (no list append)
- @tf.function for training loops
- Cached shortest paths (precomputed at reset)
- Direct NumPy slicing instead of list comprehensions