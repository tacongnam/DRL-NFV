## Phân tích nguyên nhân chính

**1. Fallback logic làm nhiễu reward signal**

Trong `_execute_plan`, khi LL agent fail, hệ thống fallback sang BestFit nhưng `R_LL_override = None` — tức là vẫn tính reward dương qua `_compute_ll_reward`. Agent học rằng "fail cũng được reward" nên không có áp lực cải thiện placement thực sự:

```python
# strategy/hrl.py ~line 295
success, rewards, score, plan, plan_source, Z_ll_fail = self._execute_plan(...)

# Nhưng reward LL chỉ bị penalty khi plan_source == "fail"
reward = R_LL if plan_source in ("ll", "fail") else -config.HRL_PENALTY_DROP
```

Khi `plan_source == "fallback"` (LL fail, BestFit thành công), reward vẫn là `R_LL` dương — agent không bị penalize đúng mức.

**2. `use_greedy` quá cao ở đầu training**

```python
use_greedy = np.random.random() < max(0.05, 1.0 - progress * 0.95)
```

Ở episode đầu (`progress ≈ 0`), xác suất dùng greedy gần 100%. LL agent học từ BestFit thay vì tự explore, dẫn đến **imitation learning collapse** — agent chỉ học copy BestFit nhưng lại tệ hơn vì thiếu context.

**3. Auxiliary head làm nhiễu VGAE gradient**

Theo `changes_220526.md`, thêm auxiliary regression head để VGAE học placement-useful features. Nhưng trong code, `lam_eff` được scale động:

```python
lam_eff = self.lambda_aux * vgae_loss / (vgae_loss + aux + 1e-7)
```

Khi `vgae_loss` lớn (early training), `lam_eff ≈ lambda_aux * 1.0`, auxiliary loss dominate và VGAE encode resource features thay vì topology — làm Z_t kém chất lượng cho routing decisions.

**4. Alpha/beta warm-up chưa thực sự được implement**

`changes_220526.md` đề cập warm-up cho alpha/beta trong LowLevelAgent, nhưng trong `_compute_ll_reward` vẫn dùng fixed constants:

```python
base = config.HRL_R_BASE_LL + config.HRL_LL_ALPHA * (time_rem / tMax) - config.HRL_LL_BETA * cost_norm
```

`HRL_LL_ALPHA = 0.5`, `HRL_LL_BETA = 1.0` — fixed, không warm-up. Agent bị penalize cost nặng ngay từ đầu khi chưa học được cách accept request trước đã.

---

## Hướng fix ưu tiên

**Fix 1 — Penalty đúng khi fallback:**
```python
# Trong train loop, khi plan_source == "fallback"
reward = -config.HRL_PENALTY_DROP * 0.5  # nhẹ hơn fail nhưng vẫn âm
```

**Fix 2 — Giảm greedy rate:**
```python
use_greedy = np.random.random() < max(0.05, 0.3 * (1.0 - progress))
# Chỉ dùng greedy 30% ở đầu, giảm nhanh
```

**Fix 3 — Warm-up beta theo progress trong reward:**
```python
def _compute_ll_reward(self, env_rewards, sfc, current_time, Z_t, vnf_feat, Z_next=None, progress=0.0):
    beta_eff = config.HRL_LL_BETA * min(1.0, progress * 2)  # warm up trong 50% đầu
    base = config.HRL_R_BASE_LL + config.HRL_LL_ALPHA * (time_rem / tMax) - beta_eff * cost_norm
```

**Fix 4 — Clamp auxiliary weight:**
```python
lam_eff = min(self.lambda_aux, self.lambda_aux * vgae_loss / (vgae_loss + aux + 1e-7))
```

Vấn đề cốt lõi là **reward signal bị nhiễu ở 3 điểm** (fallback, greedy dominance, beta penalty sớm) khiến agent không học được policy tốt hơn baseline dù architecture đã đúng hướng.