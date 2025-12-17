# GenAI-DRL Implementation Changelog

## Tổng quan

Tích hợp **Variational Autoencoder (VAE)** vào hệ thống DRL để thay thế priority-based DC selection bằng GenAI-based selection. Theo paper "GenAI Assistance for Deep Reinforcement Learning-based VNF Placement and SFC Provisioning in 5G Cores".

## Kiến trúc mới

```
┌─────────────────────────────────────────────────────────┐
│                   GenAI-DRL System                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Phase 1: Data Collection               │  │
│  │  (Random DC Selection + Pre-trained DRL)         │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Phase 2: GenAI Training                │  │
│  │  • VAE: Current_State → Next_State               │  │
│  │  • Value Network: Latent z → DC Score            │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Phase 3: GenAI-DRL Runtime             │  │
│  │  1. Extract all DC states                        │  │
│  │  2. Encoder → Latent representations             │  │
│  │  3. Value Network → DC scores                    │  │
│  │  4. Select max score DC                          │  │
│  │  5. DRL performs action on selected DC           │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Các file mới được tạo

### 1. GenAI Core Module

#### `genai/vae_model.py`
- **VAEEncoder**: Nén DC state thành latent vector z
  - Input: DC state (resources + VNF info + SFC info)
  - Output: z_mean, z_log_var
  - Architecture: FC(64) → FC(48) → Latent(32)

- **VAEDecoder**: Tái tạo next state từ latent z
  - Input: Latent z
  - Output: Predicted next DC state
  - Architecture: FC(48) → FC(64) → Output
  - **Chỉ dùng trong training**

- **ValueNetwork**: Đánh giá importance của DC
  - Input: Latent z
  - Output: Scalar value score
  - Architecture: FC(32) → FC(16) → Output(1)

- **VAEModel**: Wrapper quản lý toàn bộ module
  - Integrated training cho VAE và Value Network
  - Loss functions:
    - VAE: Reconstruction + 0.1 * KL Divergence
    - Value: MSE
  - Inference: predict_dc_values()

#### `genai/dc_state_observer.py`
- **DCStateObserver**: Trích xuất DC state cho GenAI
  - `get_dc_state()`: Extract features của 1 DC
    - Resources: CPU, RAM, Storage (normalized)
    - VNF counts: Installed, Idle per type
    - SFC info: Source count, min E2E remaining, BW needs
  - `get_all_dc_states()`: Batch processing cho tất cả DC
  - `calculate_dc_value()`: Tính ground truth value cho training
    - Factor 1: Capability to serve requests (urgency-weighted)
    - Factor 2: Resource availability
    - Factor 3: Source DC bonus

#### `genai/dc_selector.py`
- **VAEDCSelector**: Runtime DC selection
  - `select_dc()`: Chọn DC có value cao nhất
  - `get_dc_ranking()`: Ranking tất cả DC theo value

#### `genai/trainer.py`
- **VAETrainer**: Training pipeline
  - Dataset management: deque(maxlen=50000)
  - `collect_transition()`: Thu thập (current_state, next_state, value)
  - `train_vae()`: Train VAE với reconstruction + KL loss
  - `train_value_network()`: Train Value Network (encoder frozen)
  - Save/Load model weights

### 2. Environment Integration

#### `environment/gym_env_genai.py`
- **VAEEnv**: Environment với GenAI DC selection
  - Modes:
    - `data_collection_mode=True`: Random DC selection
    - `data_collection_mode=False`: GenAI selection
  - `_update_dc_order()`: GenAI ranking thay vì priority points
  - `get_dc_transitions()`: Extract transitions cho training
  - Backward compatible với original environment

### 3. Training & Evaluation Scripts

#### `runners/collect_data.py`
- Pipeline thu thập dữ liệu:
  1. Load pre-trained DRL agent
  2. Run episodes với random DC selection
  3. Collect (DC_current, DC_next, value) transitions
  4. Train VAE + Value Network
  5. Save GenAI model
- Default: 100 episodes, ~50K samples

#### `runners/train_genai.py`
- Training GenAI-DRL:
  1. Load trained GenAI model
  2. Initialize VAEEnv với GenAI DC selection
  3. Train DRL với GenAI-selected DCs
  4. Save as `models/genai_{WEIGHTS_FILE}`

#### `runners/evaluate_genai.py`
- Evaluation với GenAI-DRL
- Same experiments như DRL baseline:
  - Experiment 1: Performance per SFC type
  - Experiment 2: Scalability analysis
- Output: `fig/genai_result_*.png`

### 4. Updated Scripts

#### `scripts.py`
- Added commands:
  - `python scripts.py collect`: Data collection
  - `python scripts.py train --mode genai`: Train GenAI-DRL
  - `python scripts.py eval --mode genai`: Evaluate GenAI-DRL
- Interactive menu updated với GenAI options

## Workflow sử dụng

### Baseline DRL (existing)
```bash
python scripts.py train              # Train DRL
python scripts.py eval               # Evaluate DRL
```

### GenAI-DRL (new)
```bash
# Step 1: Train baseline DRL first (if not done)
python scripts.py train

# Step 2: Collect data cho GenAI
python scripts.py collect

# Step 3: Train GenAI-DRL
python scripts.py train --mode genai

# Step 4: Evaluate
python scripts.py eval --mode genai
```

## State Dimension

**DC State Vector** (shape: `state_dim`):
- Resources (3): CPU ratio, RAM ratio, Storage ratio
- VNF Installed counts (6): Per VNF type, normalized
- VNF Idle counts (6): Per VNF type, normalized
- SFC Info (3):
  - Source SFC count (normalized)
  - Min remaining time (normalized)
  - Total BW need (normalized)

**Total: 3 + 6 + 6 + 3 = 18 features**

## Thay đổi về hiệu suất

### Ưu điểm GenAI-DRL:
1. **Better DC Selection**: VAE học được representation tốt hơn của DC state
2. **Learned Priority**: Value Network học từ data thực tế, không hard-code
3. **Generalization**: VAE latent space giúp generalize cho unseen configurations
4. **End-to-End Learning**: DC selection và VNF placement được học cùng nhau

### Trade-offs:
1. **Training Time**: Cần 2 giai đoạn training (data collection + GenAI-DRL)
2. **Model Complexity**: Thêm 3 neural networks (Encoder, Decoder, Value)
3. **Memory**: Dataset buffer ~50K samples
4. **Inference**: Thêm forward pass qua VAE + Value Network mỗi time step

## Optimization Notes

### 1. Data Collection
- Random DC selection để exploration
- Sử dụng pre-trained DRL để quality actions
- Thu thập ~50K samples (~100 episodes)

### 2. VAE Training
- Beta = 0.1 cho KL loss (balance reconstruction vs regularization)
- Latent dim = 32 (balance representation vs complexity)
- Epochs = 50, batch = 64

### 3. Value Network Training
- Freeze encoder weights
- MSE loss với calculated values
- Epochs = 30, batch = 64

### 4. Runtime Inference
- Encoder: inference mode (chỉ lấy mean, không sample)
- Batch processing cho tất cả DC
- Efficient numpy operations

## File Structure

```
DRL/
├── genai/                          # NEW: GenAI module
│   ├── __init__.py
│   ├── vae_model.py               # VAE + Value Network
│   ├── dc_state_observer.py      # State extraction
│   ├── dc_selector.py             # DC selection logic
│   └── trainer.py                 # Training pipeline
│
├── environment/
│   ├── gym_env.py                 # Original DRL env
│   └── gym_env_genai.py           # NEW: GenAI-enabled env
│
├── runners/
│   ├── train.py                   # Original DRL training
│   ├── evaluate.py                # Original DRL eval
│   ├── collect_data.py     # NEW: Data collection
│   ├── train_genai.py             # NEW: GenAI-DRL training
│   └── evaluate_genai.py          # NEW: GenAI-DRL eval
│
├── models/
│   ├── sfc_dqn.weights.h5        # DRL weights
│   ├── genai_model_encoder.weights.h5    # NEW
│   ├── genai_model_decoder.weights.h5    # NEW
│   ├── genai_model_value.weights.h5      # NEW
│   └── genai_sfc_dqn.weights.h5          # NEW
│
└── scripts.py                     # UPDATED: Main entry point
```

## Dependencies

Không cần thêm dependencies mới. Sử dụng:
- TensorFlow/Keras (existing)
- NumPy (existing)
- Gymnasium (existing)

## Testing

### Validate GenAI Module
```python
from genai.vae_model import VAEModel
from genai.dc_state_observer import DCStateObserver

state_dim = DCStateObserver.get_state_dim()
model = VAEModel(state_dim, latent_dim=32)

# Test forward pass
import numpy as np
dummy_state = np.random.rand(4, state_dim).astype(np.float32)
values = model.predict_dc_values(dummy_state)
print(f"DC values: {values}")  # Should output 4 values
```

### Compare DRL vs GenAI-DRL
```bash
# Train both
python scripts.py train
python scripts.py collect
python scripts.py train --mode genai

# Compare
python scripts.py eval
python scripts.py eval --mode genai

# Check fig/ directory for comparison plots
```

## Performance Expectations

Based on paper results, GenAI-DRL should show:
- **Acceptance Ratio**: +5-15% improvement
- **E2E Delay**: Similar or slightly better
- **Throughput**: +10-20% improvement
- **Scalability**: Better performance với nhiều DCs (6-8)

## Future Improvements

1. **Online Learning**: Update GenAI model trong runtime
2. **Multi-objective Value**: Combine multiple objectives trong value function
3. **Attention Mechanism**: Add attention trong encoder
4. **Curriculum Learning**: Progressive training với complexity tăng dần
5. **Transfer Learning**: Pre-train GenAI trên synthetic data

## References

Paper: "GenAI Assistance for Deep Reinforcement Learning-based VNF Placement and SFC Provisioning in 5G Cores" (arXiv:2411.12851v1)

Key concepts implemented:
- Section IV.A: Generative AI Assistance Model (VAE)
- Section IV.B: Deep Reinforcement Learning Module (DQN)
- Section IV.C: Algorithm 1 - SFC Provisioning via GenAI-DRL