#!/usr/bin/env python3
"""
Quick test to verify everything works
Run this first before full training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("="*70)
print("QUICK TEST - Verifying Installation")
print("="*70)

print("\n1. Testing imports...")
try:
    from config import *
    print("   ✓ config")
    from utils import *
    print("   ✓ utils")
    from env.sfc_environment import SFCEnvironment
    print("   ✓ SFCEnvironment")
    from env.dqn_model import DQNModel, ReplayMemory
    print("   ✓ DQNModel")
    import tensorflow as tf
    print("   ✓ tensorflow")
    import gymnasium as gym
    print("   ✓ gymnasium")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\nRun: pip install -r requirements.txt")
    exit(1)

print("\n2. Testing environment creation...")
try:
    env = SFCEnvironment(num_dcs=4)
    state, _ = env.reset()
    print(f"   ✓ Environment created")
    print(f"   ✓ State shapes: {state['state1'].shape}, {state['state2'].shape}, {state['state3'].shape}")
except Exception as e:
    print(f"   ✗ Environment failed: {e}")
    exit(1)

print("\n3. Testing model creation...")
try:
    model = DQNModel(len(VNF_TYPES), len(SFC_TYPES))
    print(f"   ✓ Model created")
    print(f"   ✓ Total params: {model.model.count_params():,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    exit(1)

print("\n4. Testing forward pass...")
try:
    q_values = model.predict(state)
    print(f"   ✓ Forward pass works")
    print(f"   ✓ Q-values shape: {q_values.shape}")
    print(f"   ✓ Sample Q-values: {q_values[:3]}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    exit(1)

print("\n5. Testing environment step...")
try:
    action = 0
    next_state, reward, done, _, info = env.step(action)
    print(f"   ✓ Step executed")
    print(f"   ✓ Reward: {reward}, Done: {done}")
except Exception as e:
    print(f"   ✗ Step failed: {e}")
    exit(1)

print("\n6. Testing training update...")
try:
    from utils import generate_sfc_requests
    
    env.reset()
    requests = generate_sfc_requests()
    env.add_requests(requests)
    print(f"   ✓ Generated {len(requests)} SFC requests")
    
    memory = ReplayMemory(1000)
    
    state = env._get_state()
    for _ in range(10):
        action = model.predict(state).argmax()
        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    
    print(f"   ✓ Collected {len(memory)} experiences")
    
    if len(memory) >= 5:
        states, actions, rewards, next_states, dones = memory.sample(5)
        loss = model.train_on_batch(states, actions, rewards, next_states, dones)
        print(f"   ✓ Training step successful, loss: {loss:.4f}")
    
except Exception as e:
    print(f"   ✗ Training update failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run:")
print("  - python verify_architecture.py  (detailed architecture check)")
print("  - python main.py                 (start training)")
print("  - python tests.py                (evaluate trained model)")
print("="*70)