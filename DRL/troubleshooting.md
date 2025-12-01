# Troubleshooting Guide

## Common Errors & Solutions

### 1. CUDA Error / GPU Not Available

**Error:**
```
failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
```

**Solution:**
Code đã được configure để chạy trên CPU. Nếu vẫn gặp lỗi:

```python
# Add to top of script
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### 2. KerasTensor Error in Dueling DQN

**Error:**
```
ValueError: A KerasTensor cannot be used as input to a TensorFlow function
```

**Cause:** Cannot use `tf.reduce_mean()` directly in Functional API

**Solution:** ✅ Already fixed! Using `layers.Lambda` wrapper:
```python
mean_advantage = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantages)
output = layers.Add()([value, layers.Subtract()([advantages, mean_advantage])])
```

### 3. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'config'
```

**Solution 1:** Run from project root directory:
```bash
cd /path/to/DRL-NFV/DRL
python verify_architecture.py
```

**Solution 2:** Add project to Python path:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

**Solution 3:** Use setup script:
```bash
python -c "import setup_environment"
python verify_architecture.py
```

### 4. Gymnasium Not Found

**Error:**
```
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution:**
```bash
pip install gymnasium
```

Or in Kaggle/Colab:
```python
!pip install gymnasium
```

### 5. Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:** Reduce batch size and memory:
```python
# In config.py
TRAINING_CONFIG = {
    'batch_size': 32,        # Down from 64
    'memory_size': 50000,    # Down from 100000
    # ...
}
```

### 6. Training Too Slow

**Issue:** Training takes > 6 hours

**Solutions:**

**A. Use fewer updates for testing:**
```python
TRAINING_CONFIG['num_updates'] = 50  # Instead of 350
TRAINING_CONFIG['episodes_per_update'] = 10  # Instead of 20
```

**B. Enable GPU (if available):**
```python
# Remove this line:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Check GPU:
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

**C. Reduce training steps per update:**
```python
# In main.py, change:
for _ in range(min(num_train_steps, 20)):  # Instead of 50
```

### 7. Model Not Learning

**Symptoms:**
- Acceptance ratio stays < 40%
- Loss not decreasing
- Random-like behavior

**Diagnostic Checklist:**

```python
# Check 1: Replay memory has enough samples
print(f"Memory size: {len(trainer.memory)}")  # Should be > 1000

# Check 2: Epsilon is decaying
print(f"Epsilon: {trainer.epsilon}")  # Should decrease over time

# Check 3: Receiving varied rewards
# Look for mix of +2.0, -1.5, -1.0, -0.5

# Check 4: Q-values are updating
# Loss should gradually decrease
```

**Solutions:**

```python
# A. Increase learning rate
TRAINING_CONFIG['learning_rate'] = 0.0005  # Up from 0.0001

# B. Slower epsilon decay
TRAINING_CONFIG['epsilon_decay'] = 0.998  # Up from 0.995

# C. Larger batch size
TRAINING_CONFIG['batch_size'] = 128  # Up from 64

# D. More training steps
TRAINING_CONFIG['num_updates'] = 500  # Up from 350
```

### 8. File Not Found Errors

**Error:**
```
FileNotFoundError: checkpoints/best_model.weights.h5
```

**Solution:** Create directory first:
```python
import os
os.makedirs('checkpoints', exist_ok=True)
```

Or in main.py (already included):
```python
def main():
    import os
    os.makedirs('checkpoints', exist_ok=True)
    # ...
```

### 9. Matplotlib Display Issues

**Error:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solution:** Save plots instead of showing:
```python
plt.savefig('plot.png')
plt.close()
# Don't use plt.show()
```

Already implemented in code! ✅

### 10. Weird Acceptance Ratios (> 100% or negative)

**Cause:** Division by zero or incorrect counting

**Check:**
```python
print(f"Accepted: {env.total_accepted}")
print(f"Dropped: {env.total_dropped}")
print(f"Generated: {env.total_generated}")
```

**Debug in environment:**
```python
# In env/sfc_environment.py::get_acceptance_ratio()
def get_acceptance_ratio(self):
    if self.total_generated == 0:
        return 0.0
    ratio = self.total_accepted / self.total_generated
    return np.clip(ratio, 0.0, 1.0)  # Ensure [0, 1] range
```

## Platform-Specific Issues

### Kaggle

```python
# Install dependencies
!pip install gymnasium

# Run with CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Reduce resources
TRAINING_CONFIG['num_updates'] = 50  # Quick test
```

### Google Colab

```python
# Mount Drive (optional - to save checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Install
!pip install gymnasium

# Run
!python verify_architecture.py
```

### Local Machine

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
python verify_architecture.py
```

## Debugging Tips

### Enable Verbose Logging

```python
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# In training loop
print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")
```

### Check Model Summary

```python
model = DQNModel(len(VNF_TYPES), len(SFC_TYPES))
model.model.summary()

# Should show:
# - 3 input layers
# - Concatenate layer
# - Attention layer
# - Value and Advantage streams
```

### Test Single Step

```python
env = SFCEnvironment(num_dcs=4)
state, _ = env.reset()

# Test state shape
print(f"State 1: {state['state1'].shape}")  # Should be (14,)
print(f"State 2: {state['state2'].shape}")  # Should be (78,)
print(f"State 3: {state['state3'].shape}")  # Should be (60,)

# Test action
action = 0  # Allocate NAT
next_state, reward, done, _, info = env.step(action)
print(f"Reward: {reward}, Done: {done}")
```

### Verify Imports

```python
# test_imports.py
try:
    from config import *
    print("✓ config imported")
    
    from utils import *
    print("✓ utils imported")
    
    from env.sfc_environment import SFCEnvironment
    print("✓ environment imported")
    
    from env.dqn_model import DQNModel
    print("✓ model imported")
    
    print("\n✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
```

## Getting Help

If you're still stuck:

1. **Check error message carefully** - Often tells you exactly what's wrong
2. **Run `verify_architecture.py`** - Validates setup
3. **Test with minimal config** - Use 10 updates, 5 episodes
4. **Check file structure:**
   ```
   DRL/
   ├── config.py
   ├── utils.py
   ├── main.py
   ├── tests.py
   ├── verify_architecture.py
   └── env/
       ├── __init__.py
       ├── sfc_environment.py
       └── dqn_model.py
   ```

5. **Print intermediate values** - Add debug prints liberally

## Quick Health Check

Run this to verify everything works:

```bash
# 1. Test imports
python -c "from env.sfc_environment import SFCEnvironment; from env.dqn_model import DQNModel; print('✅ Imports OK')"

# 2. Verify architecture
python verify_architecture.py

# 3. Quick training test
python -c "
from main import SFCTrainer
trainer = SFCTrainer(num_dcs=4)
reward, acc = trainer.train_episode()
print(f'✅ Episode completed: reward={reward:.2f}, acc={acc:.3f}')
"
```

Expected output: All ✅ checks pass!