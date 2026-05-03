# HRL-VGAE CODE REVIEW - Phát hiện lỗi sau refactoring

## 1. LỖI CRITICAL

### LỖI A: Cache precision inconsistency
**File**: strategy/hrl.py, lines 132, 145, 165
```python
# HRL custom routing caches:
key = (t_start, t_end, round(bw, 1))  # ← Precision = 1

# vs RoutingMixin (inherited by other strategies):
key = (t_start, t_end, round(bw, 2))  # ← Precision = 2
```
**Vấn đề**: 
- Khi request bw=50.05, round(bw,1)=50.1 vs round(bw,2)=50.05
- HRL sẽ miss cache từ RoutingMixin nếu có sharing
- Precision không consistency → cache trở nên không đáng tin

**Impact**: MAJOR - Cache inefficiency, potential topology mismatch

---

### LỖI B: plan=None nhưng vẫn access plan.get()
**File**: strategy/hrl.py, line 628
```python
success, rewards, score = self._execute_plan(plan, selected_sfc, t, snap)

if success:
    accepted += 1
    node_cost = sum(...for k, v in plan.get("nodes", {}).items()...)
    # ↑ plan có thể None nếu _execute_plan() retry fallback thành công 
    # nhưng fallback không return plan object
```
**Vấn đề**:
- `_execute_plan()` return `(success, rewards, score)` - không return plan
- `plan` từ line 632 (ngoài hàm) có thể None
- `.get("nodes", {})` sẽ crash nếu plan=None

**Impact**: CRITICAL - AttributeError at runtime (eval phase)

---

### LỖI C: R_LL_override logic không rõ ràng (training phase)
**File**: strategy/hrl.py, lines 437, 488-509
```python
# Greedy phase:
R_LL_override = 1.5

# Non-greedy phase:
R_LL_override = None if plan is not None else 1.0

# Compute:
if R_LL_override is not None:
    R_LL = R_LL_override
else:
    R_LL = self._compute_ll_reward(...)
```
**Vấn đề**:
- 3 cases khác nhau (1.5, 1.0, None) - không có comment giải thích logic
- Greedy vs non-greedy reward weights không symmetric
- Nếu plan=None (fallback failed), R_LL_override=1.0 nhưng success=False
- Khi failure, penalty được apply (R_LL=-PENALTY_DROP) nên override không được dùng

**Impact**: MAJOR - Training signal bị confusing

---

## 2. LỖI MAJOR - DESIGN

### LỖI D: Greedy phase trajectory rebuild không off-policy correct
**File**: strategy/hrl.py, lines 437-465
```python
if use_greedy:
    plan = self._get_best_fit().get_placement(...)
    self._ll_traj = []  # Clear
    R_LL_override = 1.5
    
    if plan is not None:
        # Rebuild trajectory từ best_fit plan
        # Nhưng best_fit không từ LL agent → action không match mask
```
**Vấn đề**:
- Best_fit plan không tuân theo LL agent's mask
- Trajectory rebuild có `valid_indices` từ DC mapping
- Off-policy training nhưng không có importance sampling

**Impact**: MAJOR - LL agent training signal bị lỗi

---

### LỖI E: Time_ratio không consistent giữa training/eval
**File**: strategy/hrl.py, line 498 (training) vs line 523-633 (eval)
```python
# Training:
time_ratio = min(1.0, (t - arrival_time) / delay_max)
R_HL = [BASE_AR_REWARD + time_ratio, -cost_norm]

# Eval:
# Không tính time_ratio! Chỉ tính cost
```
**Vấn đề**:
- Training reward = reward(time_ratio + cost)
- Eval không dùng time_ratio → agent predict sai

**Impact**: MAJOR - Train/eval mismatch

---

## 3. LỖI MINOR

### LỖI F: Cache key round precision (RoutingMixin)
- `round(bw, 2)` có thể miss nearby values
- `round(bw, 1)` quá coarse
- **Recommend**: Use fixed precision `round(bw, 2)` nhất quán

---

### LỖI G: Fallback trajectory không append (line 488)
```python
else:
    plan = self.get_placement(...)
    R_LL_override = None if plan is not None else 1.0
    # Nếu plan=None, trajectory vẫn là từ get_placement() attempt
    # Không rebuild từ fallback!
```
**Impact**: MINOR - Inconsistent trajectory source

---

## 4. CHƯA IMPLEMENT ĐƯỢC (Nhưng required)

### Lỗi H: Multi-objective aggregation incomplete
- Pareto ranking: NOT IMPLEMENTED
- Crowding distance: NOT IMPLEMENTED
- HL agent chỉ epsilon-greedy, không multi-objective selection

---

## TỔNG HỢP ĐỘ CIDRTICAL

| Lỗi | Mức | Type | Fix |
|-----|-----|------|-----|
| A (Cache precision) | MAJOR | Design | Unify round(bw, 2) |
| B (plan=None access) | **CRITICAL** | Bug | Return plan from _execute_plan() |
| C (R_LL_override confusing) | MAJOR | Logic | Clarify 3 cases |
| D (Greedy trajectory) | MAJOR | Training | Fix off-policy |
| E (Time_ratio missing eval) | MAJOR | Train/eval | Add time_ratio in eval |
| F (Cache precision again) | MINOR | Efficiency | Standardize |
| G (Fallback traj) | MINOR | Consistency | Document or fix |
| H (Pareto) | FEATURE | Not implemented | Priority later |

---

## PHÁT HIỆN MỚI SAU REFACTORING

✅ Fixed:
- _execute_plan() consolidation ✓
- Graph symmetry A[i,j] = A[j,i] ✓
- Time_ratio bounded ✓
- Code duplication eliminated ✓

❌ Remaining:
- Cache precision inconsistency
- plan=None access bug
- Greedy trajectory logic
- Train/eval reward mismatch
