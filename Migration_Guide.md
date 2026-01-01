# Migration Guide: Per-SFC-Type → Hybrid Chain Pattern Encoding

## Problem
Original implementation assumes **fixed 6 SFC types** with per-type feature encoding (152 dimensions). Real datasets have **2500+ unique SFC chains** → cannot encode each as separate type, but simple VNF aggregation loses too much information (46 dimensions).

## Solution
**Hybrid Encoding: Top-K Chain Patterns + VNF Aggregation**

---

---

## Dimension Comparison Table

| Approach | s1 | s2 | s3 | Total | Info Quality |
|----------|----|----|----|----|--------------|
| **Original** (6 SFC types, 6 VNF types) | 14 | 78 | 60 | **152** | ⭐⭐⭐ Fixed types only |
| **Simple Aggregate** (10 VNF types) | 22 | 10 | 14 | **46** | ⭐ 70% info loss |
| **Hybrid Pattern** (10 VNF types) | 22 | 61 | 99 | **182** | ⭐⭐⭐⭐ Dynamic + order-aware |

**Why Hybrid is Better:**
1. ✅ **More features** than original (182 vs 152)
2. ✅ **Captures chain order** (not just VNF presence)
3. ✅ **Dynamic**: Adapts to any dataset (1 to 2500+ chains)
4. ✅ **Efficient**: Only encodes top-K patterns (not all 2500+)
5. ✅ **Generalizable**: Model learns from common patterns

## Dimension Comparison Table

---

## Chain Pattern Encoding Details

### Structure
Each chain pattern encoded as `(4 + NUM_VNF_TYPES + 3)` features:

1. **Sequence** (4 features): VNF indices in order, padded with -1
   ```python
   chain = [0, 1]  # NAT → FW
   seq = [0/10, 1/10, -1, -1] = [0.0, 0.1, -1, -1]  # Normalized
   ```

2. **Presence** (NUM_VNF_TYPES features): Binary indicator for each VNF
   ```python
   presence = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # NAT and FW present
   ```

3. **Statistics** (3 features):
   - `count`: Proportion of requests with this pattern (0-1)
   - `avg_bw`: Average bandwidth requirement (Mbps, normalized)
   - `avg_rem_time`: Average remaining time (ms, normalized)

### Example
```python
# Chain [0, 1] appears 15 times out of 100 requests
# Average: BW=2.5 Mbps, Remaining=40ms

encoded = [
    0.0, 0.1, -1.0, -1.0,              # Sequence
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0,      # Presence (10 VNF types)
    0.15, 0.0025, 0.4                  # Stats (normalized)
]
# Total: 17 features per chain pattern
```

### Aggregation Levels
- **DC-level (s2)**: Top-3 chain patterns from requests originating at this DC
- **Global (s3)**: Top-5 chain patterns across all active requests

### Benefits
✅ Captures chain order (not just VNF presence)  
✅ Dynamic: Works with any number of unique chains  
✅ Efficient: Only encodes most common patterns  
✅ Informative: More features than original (182 vs 152)

---

## Implementation

### Observer (`envs/observer.py`)

**New Helper Functions:**
```python
@staticmethod
def _encode_chain_pattern(chain, max_length=4):
    """Encode single chain into fixed-size representation"""
    # Chain sequence (padded)
    chain_seq = np.full(max_length, -1, dtype=np.float32)
    for i, vnf in enumerate(chain[:max_length]):
        chain_seq[i] = vnf / config.NUM_VNF_TYPES
    
    # VNF presence
    vnf_presence = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
    for vnf in chain:
        if vnf < config.NUM_VNF_TYPES:
            vnf_presence[vnf] = 1.0
    
    return np.concatenate([chain_seq, vnf_presence])

@staticmethod
def _aggregate_chain_stats(active_reqs, max_top_chains=5):
    """Aggregate top-K chain patterns with statistics"""
    from collections import Counter
    
    chain_counter = Counter(tuple(r.chain) for r in active_reqs)
    top_chains = chain_counter.most_common(max_top_chains)
    
    result = []
    for chain_tuple, count in top_chains:
        pattern = Observer._encode_chain_pattern(list(chain_tuple))
        stats = compute_stats(chain_tuple, active_reqs)  # BW, time
        result.append(np.concatenate([pattern, stats]))
    
    # Pad to max_top_chains with zeros
    while len(result) < max_top_chains:
        result.append(np.zeros(feature_size))
    
    return np.concatenate(result)
```

**Updated Observation:**
```python
@staticmethod
def get_drl_observation(dc, sfc_manager):
    # s1: DC state (unchanged)
    s1 = [cpu, ram, installed_vnfs, idle_vnfs]
    
    # s2: DC demand = VNF counts + top-3 chains at this DC
    vnf_demand = compute_vnf_demand_at_dc(dc)
    dc_chains = [r for r in active_reqs if r.source == dc.id]
    chain_patterns_dc = _aggregate_chain_stats(dc_chains, max_top_chains=3)
    s2 = np.concatenate([vnf_demand, chain_patterns_dc])
    
    # s3: Global = stats + VNF demand + top-5 chains globally
    global_stats = [count, avg_rem, avg_bw, drop_rate]
    global_vnf = compute_global_vnf_demand()
    chain_patterns_global = _aggregate_chain_stats(active_reqs, max_top_chains=5)
    s3 = np.concatenate([global_stats, global_vnf, chain_patterns_global])
    
    return (s1, s2, s3)
```

### DQN Model (`agents/dqn_model.py`)

```python
def build_q_network():
    V = config.NUM_VNF_TYPES
    chain_feature_size = 4 + V + 3  # Seq + Presence + Stats
    
    input_1_shape = (2 + 2*V,)
    input_2_shape = (V + 3*chain_feature_size,)  # VNF + 3 chains
    input_3_shape = (4 + V + 5*chain_feature_size,)  # Stats + VNF + 5 chains
    
    # ... rest of model architecture
```

---

## JSON Format Changes

**Before (Paper Format):**
```json
{
  "R": [
    {
      "type": "CloudGaming",
      "source": 0,
      "destination": 2,
      "bandwidth": 4,
      "max_delay": 80
    }
  ]
}
```

**After (Real Dataset Format):**
```json
{
  "F": [
    {"c_f": 1.2, "r_f": 1.0, "h_f": 0.8, "d_f": {"0": 0.3}}
  ],
  "R": [
    {
      "T": 0,
      "st_r": 2,
      "d_r": 4,
      "F_r": [0, 1],
      "b_r": 1.5,
      "d_max": 2.0,
      "type": "Optional_Label"
    }
  ]
}
```

**Key Differences:**
- `F_r`: VNF chain as **indices**, not SFC type name
- `type`: Optional field for post-analysis only
- Each request specifies its own `b_r` and `d_max`

---

## Usage Example

```python
# Load data
reader = Read_data('data/test.json')
vnf_specs = reader.get_F()
requests = reader.get_R()

# Update config BEFORE creating models
config.update_vnf_specs(vnf_specs)
config.ACTION_SPACE_SIZE = config.get_action_space_size()

# Now create environment/agent
env = DRLEnv(...)
agent = Agent()  # Uses config.ACTION_SPACE_SIZE internally
```

---

## Testing

```bash
# Verify shapes with small dataset
python runners/train_drl.py  # Should print:
#   VNF types: 2
#   Action space: 5  (2×2+1)

# Check observation dimensions
# s1: (2 + 2×2,) = (6,)
# s2: (2,)
# s3: (4 + 2,) = (6,)
```

---

## Benefits

1. **Scalability**: Handles any number of unique SFC chains (1 to 2500+)
2. **Generalization**: Model learns VNF-level patterns, not SFC-specific
3. **Smaller State Space**: `O(NUM_VNF_TYPES)` vs `O(NUM_SFC_TYPES × NUM_VNF_TYPES)`
4. **Dynamic**: No code changes needed for different datasets

---

## Limitations

- Cannot do per-SFC-type analysis (unless `type` field is populated in JSON)
- Model treats all requests equally (no type-specific prioritization)
- Solution: Add optional `priority` field in JSON if needed