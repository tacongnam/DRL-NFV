# DRL-NFV: Deep Reinforcement Learning for NFV Placement

## 1. Problem Definition

### 1.1 Input

| Symbol | Description |
|---|---|
| $G = (N, L)$ | Physical network: nodes $N$ (Data Centers + Switches), links $L$ |
| $f \in F$ | VNF type with resource demand $r_f = \{mem_f, cpu_f, ram_f\}$ and optional DC affinity $d_f$ |
| $req_k \in R$ | Request: $(arrival\_time,\ delay\_max,\ start\_node,\ end\_node,\ [f_1,\ldots,f_m],\ bw)$ |
| $end\_time_k$ | $= arrival\_time_k + delay\_max_k$ |

**DC affinity:** If $d_f = \{-1\}$, the VNF can be placed on any DC. Otherwise restricted to listed DC names.

### 1.2 Decision Variables

- **Placement:** assignment of each VNF $f_i$ to a DC node $n \in N_{DC}$
- **Routing:** sequence of nodes forming a path between consecutive placement points

### 1.3 Constraints

**Node capacity** — for each DC $n$, resource type $k$, active timeslot $T$:
$$\sum_{f \in placed(n,T)} r_f[k] \leq cap_n[k]$$

**Link bandwidth** — for each link $l$, active timeslot $T$:
$$used\_bw_{l,T} + bw_{req} \leq cap_l$$

**Path delay:**
$$\sum_{l \in path} delay_l \leq delay\_max$$

**VNF chain ordering:** Traffic routed $start\_node \to DC(f_1) \to \cdots \to DC(f_m) \to end\_node$.

### 1.4 Timeslot Conversion

$$T = \left\lfloor \frac{t}{TIMESTEP} + 0.5 \right\rfloor, \quad TIMESTEP = 0.1$$

---

## 2. System Overview
Request arrives

│

▼

[VGAE Encoder] ── DC graph snapshot (X, A)  →  Z ∈ ℝ^{|DC| × LATENT_DIM}

│

▼

[Placer DQN] ── for each VNF in SFC: select DC (epsilon-greedy) + route path

│

▼

[Admission PPO] ── accept / reject the full plan

│

▼

[Env.step(plan)] ── validate + commit resource usage

│

▼

[Replay Buffer] ── train Placer DQN + fine-tune VGAE aux head online

**Baselines:** `GreedyFIFS`, `BestFit`, `DeadlineAwareGreedy`, `RandomFit`, `ShortestPathFirst`, `GreedyGLB`.

---

## 3. Resource Tracking

### 3.1 Node Usage

`Node.used`: `{timeslot: {resource_type: value}}` — step function semantics.

`Node.use(resource, start_T, end_T)`: fills missing timeslots by copying nearest prior snapshot, then adds demand to $[start\_T, end\_T)$.

`Node.get_min_available_resource(t_start, t_end)`: minimum available resource across all timeslots in window.

### 3.2 Link Usage

`Link.used`: `{timeslot: bw_used}` — same step-function semantics.

`Link.get_available_bandwidth(start_T, end_T)`: minimum available bandwidth across window.

### 3.3 Snapshot & Rollback

```python
snap = snapshot_network(network)   # shallow copy of used dicts
restore_network(network, snap)     # reassign used dicts on failure
```

Enables atomic rollback if any constraint is violated mid-placement.

---

## 4. VGAE Graph Encoding

### 4.1 DC Node Feature Matrix $X$ (10 features per DC)

| Index | Description |
|---|---|
| 0–2 | $avail_k / max\_cap_k$ — normalized available resource (mem, cpu, ram) |
| 3–5 | $(cap_k - avail_k) / cap_k$ — load ratio |
| 6 | M/M/1 queue pressure: $\min\!\left(\frac{\bar{\rho}}{1-\bar{\rho}} / 20,\ 1\right)$ |
| 7 | Mean link load of adjacent links |
| 8 | Normalized total available bandwidth of adjacent links |
| 9 | Degree (number of adjacent links) / 10 |

$avail_k = \min_{t \in [T_s, T_e]} (cap_k - used_k(t))$ over the request window.

### 4.2 Adjacency Matrix $A$

$$A_{ij} = \begin{cases}
1.0 & i = j \\
1 / (composite\_dist(i,j) + 1) & \text{if BW-feasible path exists} \\
0 & \text{otherwise}
\end{cases}$$

The graph prunes links with available bandwidth $< bw_{req}$, then runs `nx.shortest_path_length` with `weight="weight"` using composite edge weights.

### 4.3 Normalized Adjacency

$$\hat{A} = D^{-1/2}(A + I)D^{-1/2}$$

### 4.4 GCN Encoder

$$H^{(1)} = \text{ReLU}(\hat{A} X W_1)$$
$$\mu = \hat{A} H^{(1)} W_\mu, \quad \log\sigma^2 = \text{clip}(\hat{A} H^{(1)} W_{lv},\ -10,\ 10)$$

**Training:** $Z = \mu + \sigma \odot \epsilon,\ \epsilon \sim \mathcal{N}(0,I)$  
**Inference (deterministic):** $Z = \mu$

### 4.5 VGAE Loss

$$\mathcal{L} = \mathcal{L}_{BCE} + \beta \cdot \mathcal{L}_{KL} + \lambda \cdot \mathcal{L}_{aux}$$

$$\mathcal{L}_{BCE} = -\mathbb{E}\left[A \log \sigma(ZZ^T) + (1-A)\log(1 - \sigma(ZZ^T))\right]$$

$$\mathcal{L}_{KL} = -\frac{1}{2}\mathbb{E}\left[1 + \log\sigma^2 - \mu^2 - \sigma^2\right]$$

$$\mathcal{L}_{aux} = \|g(Z) - X\|^2_F, \quad \lambda = \lambda_0 \cdot \frac{\mathcal{L}_{VGAE}}{\mathcal{L}_{VGAE} + \mathcal{L}_{aux} + \epsilon}$$

**Online fine-tuning:** only aux head is trained (backbone frozen) at every `HRL_VGAE_FINETUNE_FREQ` steps.

---

## 5. Placer DQN

### 5.1 State Vector

$$s = \left[\underbrace{\bar{Z}}_{\text{LATENT\_DIM}} \;\Big|\; \underbrace{r_f}_{3} \;\Big|\; \underbrace{z_{prev}}_{\text{LATENT\_DIM}} \;\Big|\; \underbrace{p_{node}}_{1}\right] \in \mathbb{R}^{2 \cdot LATENT\_DIM + 4}$$

- $\bar{Z} = \text{mean}(Z,\ \text{axis}=0)$: global DC graph summary
- $r_f$: VNF resource demand (mem, cpu, ram)
- $z_{prev}$: latent vector of previously chosen DC (zeros for first VNF)
- $p_{node}$: mean M/M/1 pressure over candidate DCs

**M/M/1 node pressure:**
$$p = \frac{1}{|K|}\sum_{k} \min\!\left(\frac{\rho_k}{1-\rho_k} / 20,\ 1\right), \quad \rho_k = \frac{cap_k - avail_k}{cap_k}$$

### 5.2 Network Architecture

Policy, target, and weight networks are 3-layer MLPs:
Input(2·LATENT_DIM+4) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(MAX_DCS)

Weight net: same input → Dense(32, ReLU) → Dense(16, ReLU) → Dense(2) → Sigmoid × [2.0, 1.0]

### 5.3 Action Selection
Q = policy_net(s)

mask invalid indices to -1e9

action = argmax(Q)          # greedy

or random.choice()  # with probability epsilon

Valid indices: DC in VNF affinity set, within MAX_DCS, sufficient resources in window.

### 5.4 Reward

$$R = DRL\_R\_BASE\_LL + \alpha(1 - time\_ratio) - \beta \cdot cost\_norm$$

| Term | Formula |
|---|---|
| $time\_ratio$ | $\min(1,\ (t - arrival\_time) / delay\_max)$ |
| $cost\_norm$ | $\min(1,\ cost(vnf, dc) / max\_cost)$ |
| $\alpha, \beta$ | output of weight network $\times [2.0, 1.0]$ |

**Failure:** $R = -DRL\_PENALTY\_DROP$  
**Admission rejection of valid plan:** $R = -DRL\_PENALTY\_DROP \times 0.3$

### 5.5 TD Training

Rewards normalized by running mean/variance (window=200):

$$\hat{r} = (r - \mu_r) / (\sigma_r + \epsilon)$$

$$\text{Target} = \hat{r} + \gamma \cdot \max_{a'} Q_{target}(s', a') \cdot (1 - done)$$

$$\mathcal{L}_{DQN} = \mathbb{E}\left[(Target - Q_{policy}(s, a))^2\right]$$

$$\mathcal{L}_{W} = \mathbb{E}\left[\left(\sigma(W(s)) - \max(r,0)/(\max_r + \epsilon)\right)^2\right]$$

---

## 6. Admission Agent (PPO)

Binary accept/reject over a sliding window of $N=10$ state triplets $(G_P, G_Q, O_Q)$.

| Feature | Dim | Description |
|---|---|---|
| $G_P$ | 8 | Network state: mean avail ratio (3) + mean load (3) + M/M/1 (1) + link load (1) |
| $G_Q$ | 5 | Request: mean VNF resource (3) + bw (1) + num_vnfs/10 (1) |
| $O_Q$ | 3 | Plan quality: revenue norm (1) + cost norm (1) + R2C norm (1) |

**Architecture:** GRU(64) → Dense(32, ReLU) → Dense(out\_dim) for both actor and critic.

**Training:** PPO with GAE ($\gamma=0.99$, $\lambda=0.95$, clip $\epsilon=0.2$), updated at end of each episode once $ep \geq episodes/3$.

**Admission reward:** $R_{adm} = \min(R2C \cdot rev / 1000,\ 10)$ on success; $0$ on failure.

---

## 7. Routing

### 7.1 Composite Edge Weight

$$w_{uv} = w_{delay} \cdot \frac{delay_{uv}}{ref\_delay} + w_{bw} \cdot \exp\!\left(-\frac{avail\_bw - bw_{req}}{cap}\right) + w_{mm1} \cdot \frac{\rho/(1-\rho)}{20} + w_{hops} \cdot 1$$

Constants: $w_{delay}=0.4,\ w_{bw}=0.3,\ w_{mm1}=0.2,\ w_{hops}=0.1$.

Links with $avail\_bw < bw_{req}$ are pruned before path search.

### 7.2 Path Selection

`nx.shortest_path(G, u, v, weight="weight")`. If $u = v$, return $[u]$ immediately.

### 7.3 Delay Validation

$$\sum_{l \in path} delay_l \leq delay\_max$$

Violated → placement attempt returns `None`.

---

## 8. Training Pipeline

### 8.1 Pretraining Stage 1 — VGAE

1. For each training file, run BestFit on all requests; collect DC graph snapshots $(X, A)$ before and after each deployment.
2. Push snapshots to `ReplayBuffer` (capacity 2000).
3. Train VGAE for `vgae_epochs` epochs, batch size 32.

### 8.2 Pretraining Stage 2 — Placer Behavioral Cloning

Teacher: 70% BestFit + 30% RandomFit per episode.

1. For each request, build DC graph, encode with VGAE.
2. If teacher succeeds: extract per-VNF DC assignments as supervision.
3. If teacher fails: push negative transitions with $R = -DRL\_PENALTY\_DROP$.
4. Train placer every episode when buffer ≥ batch size.

Epsilon decays from 0.5 to 0.05 over episodes.

### 8.3 Online Training

**Per-episode loop:**
reset env; clear caches; reset admission history

pending = sort requests by arrival_time (EDF batch ordering)
for each SFC (EDF order):

epsilon = EPSILON_MAX × 0.99^(progress×100), clipped to EPSILON_MIN

Z_t, dcs = _get_z(t_start, t_end, bw, vnf_demand)   # fresh each call

snap = snapshot_network()
with prob use_greedy_rate(progress):   # max(0.05, 0.5×(1−2×progress))
    plan = BestFit; rebuild trajectory from plan
else:
    plan = get_placement(Z_t, dcs, epsilon)   # DQN

if plan is None:
    push trajectory with R = −DRL_PENALTY_DROP; continue

(accept, log_prob, value) = admission.decide(gp, gq, oq, training)
if not accept:
    push trajectory with R = −DRL_PENALTY_DROP×0.3; continue

success = env.step(plan)
if success:
    R_placer = DRL_R_BASE_LL + α(1−time_ratio) − β·cost_norm
    R_adm    = min(R2C × rev / 1000, 10)
else:
    restore_network(snap)
    R_placer = −DRL_PENALTY_DROP; R_adm = 0

push trajectory + (X, A)
train placer every 4 steps if buf ≥ DRL_BATCH_SIZE
sync target every DRL_TARGET_SYNC steps
fine-tune VGAE aux head every HRL_VGAE_FINETUNE_FREQ steps
end-of-episode: PPO update admission (if ep ≥ episodes/3 and |traj| ≥ 2)

### 8.4 Epsilon Schedule

$$\epsilon(progress) = \max(\epsilon_{min},\ \epsilon_{max} \times 0.99^{progress \times 100})$$

Constants: $\epsilon_{max}=1.0,\ \epsilon_{min}=0.05$.

### 8.5 Multi-file Training

Episodes distributed across files. Weights (placer policy, weight net, VGAE backbone) and replay buffers transferred between files. VGAE backbone frozen after first file.

---

## 9. Evaluation

1. Load model weights (placer policy + weight net + VGAE).
2. Process requests deterministically ($\epsilon = 0$).
3. Admission agent in exploit mode (argmax).
4. Failed placer placements → BestFit fallback via `execute_with_fallback`.

| Metric | Formula |
|---|---|
| Acceptance Ratio | $accepted / total$ |
| Total Cost | $\sum_{accepted} \sum_{vnf} cost(vnf, dc)$ |
| Average Cost | $total\_cost / accepted$ |
| Total Delay | $\sum_{accepted} (end\_time - arrival\_time)$ |

---

## 10. Baselines

| Strategy | Description |
|---|---|
| `GreedyFIFS` | First-In First-Served; first feasible DC |
| `BestFit` | Minimize waste: sort DCs by $\sum_k (avail_k - demand_k)$, pick smallest |
| `DeadlineAwareGreedy` | EDF priority + BestFit placement |
| `RandomFit` | Random feasible DC |
| `ShortestPathFirst` | Prioritize paths with minimum total delay |
| `GreedyGLB` | Global load balancing greedy |

All baselines use `RoutingMixin` (same composite-weight shortest path logic).

---

## 11. Key Configuration

| Parameter | Value | Description |
|---|---|---|
| `LATENT_DIM` | 10 | VGAE latent dimension |
| `MAX_DCS` | 60 | Max DC action space |
| `TIMESTEP` | 0.1 | Time quantization unit |
| `DRL_BATCH_SIZE` | 64 | Placer training batch size |
| `DRL_TARGET_SYNC` | 100 | Steps between target network sync |
| `HRL_VGAE_FINETUNE_FREQ` | 500 | Steps between VGAE aux-head fine-tune |
| `HRL_VGAE_FINETUNE_EPOCHS` | 1 | Epochs per fine-tune |
| `DRL_R_BASE_LL` | 5.0 | Base reward for successful placement |
| `DRL_PENALTY_DROP` | 5.0 | Penalty for failed placement |
| `DRL_LL_ALPHA` | 2.0 | Delay weight in reward |
| `DRL_LL_BETA` | 0.1 | Cost weight in reward |
| `EPSILON_MAX / MIN` | 1.0 / 0.05 | Exploration range |
| `ROUTING_DELAY_WEIGHT` | 0.4 | Delay component in edge weight |
| `ROUTING_BW_WEIGHT` | 0.3 | BW pressure in edge weight |
| `ROUTING_PRESSURE_WEIGHT` | 0.2 | M/M/1 pressure in edge weight |
| `ROUTING_HOP_WEIGHT` | 0.1 | Hop count in edge weight |