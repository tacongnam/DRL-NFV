# DRL-NFV: Deep Reinforcement Learning for NFV Placement

---

## 1. Problem Definition

### 1.1 Input

| Symbol | Description |
|---|---|
| $G = (N, L)$ | Physical network: nodes $N$ (Data Centers + Switches), links $L$ |
| $f \in F$ | VNF type with resource demand $r_f = \{mem_f, cpu_f, ram_f\}$ and optional DC affinity $d_f$ |
| $req_k \in R$ | Request: $(arrival\_time,\ delay\_max,\ start\_node,\ end\_node,\ [f_1,\ldots,f_m],\ bw)$ |
| $end\_time_k$ | $= arrival\_time_k + delay\_max_k$ |

**DC affinity:** If $d_f = \{-1: 0.0\}$ (wildcard), the VNF can be placed on any DC. Otherwise it is restricted to the listed DC names.

### 1.2 Decision Variables

- **Placement:** assignment of each VNF $f_i$ in a request to a DC node $n \in N_{DC}$
- **Routing:** a sequence of nodes forming a path between consecutive placement points

### 1.3 Constraints

**Node capacity** — for each DC $n$, resource type $k$, and active timeslot $T$:
$$\sum_{f \in placed(n,T)} r_f[k] \leq cap_n[k]$$

**Link bandwidth** — for each link $l$ and active timeslot $T$:
$$used\_bw_{l,T} + bw_{req} \leq cap_l$$

**Path delay:**
$$\sum_{l \in path} delay_l \leq delay\_max$$

**VNF chain ordering:** VNFs $f_1 \to f_2 \to \cdots \to f_m$ must be deployed in sequence. Traffic is routed: $start\_node \to DC(f_1) \to DC(f_2) \to \cdots \to DC(f_m) \to end\_node$.

### 1.4 Timeslot Conversion

All continuous time values are mapped to integer timeslots before resource accounting:
$$T = \left\lfloor \frac{t}{TIMESTEP} + 0.5 \right\rfloor, \quad TIMESTEP = 0.1$$

Resource usage at timeslot $T$ is tracked as a step function. When querying usage over $[T_{start}, T_{end}]$, the system looks up all recorded timeslots in range; if none exist, it falls back to the nearest earlier snapshot.

---

## 2. System Overview

DRL-NFV combines three main components:

```
Request arrives
      │
      ▼
[VGAE Encoder] ──── DC graph snapshot (X, A)
      │                      │
      │             Z ∈ ℝ^{|DC| × LATENT_DIM}
      ▼
[Placer DQN] ──── for each VNF in SFC:
      │               select DC index (epsilon-greedy)
      │               route path via RoutingMixin
      ▼
[Env.step(plan)] ── validate + commit resource usage
      │
      ▼
[Replay Buffer] ──── train Placer + fine-tune VGAE online
```

**Baselines** available for comparison: `GreedyFIFS`, `BestFit`, `DeadlineAwareGreedy`, `RandomFit`.

---

## 3. Resource Tracking (`env/network.py`)

### 3.1 Node Usage

`Node.used` is a dict `{timeslot: {resource_type: value}}`. It stores a **step function**: usage is constant between recorded timeslots.

**`Node.use(resource, start_T, end_T)`:**
- Fills all missing timeslots in $[start\_T, end\_T)$ by copying the nearest prior snapshot.
- Ensures `end_T` has an entry (to mark the release point).
- Adds resource demand to every timeslot in $[start\_T, end\_T)$.

**`Node.get_min_available_resource(t_start, t_end)`:**
Returns the minimum available resource across all timeslots in the window — the tightest bottleneck.

### 3.2 Link Usage

`Link.used` is a dict `{timeslot: bw_used}`, same step-function semantics.

**`Link.get_available_bandwidth(start_T, end_T)`:**
Returns the minimum available bandwidth across the window.

### 3.3 Snapshot & Restore

Before attempting a placement, the environment takes a shallow snapshot:

```python
snapshot_network(network)   # copies used dicts by reference
restore_network(network, snap)  # reassigns used dicts on failure
```

This enables atomic rollback if any constraint is violated mid-placement.

---

## 4. VGAE Graph Encoding (`models/model.py`, `models/pretrain.py`)

### 4.1 DC Node Feature Matrix $X$

Built in `build_dc_graph`. Each DC $i$ contributes a row of 6 features:

| Feature index | Formula | Description |
|---|---|---|
| 0–2 | $\frac{avail_k}{max\_cap_k}$ | Normalized available resource (mem, cpu, ram) |
| 3–5 | $\min\!\left(\frac{avail_k - demand_k}{max\_cap_k},\ 1\right)$ if $slack > 0$ else $0$ | Normalized resource slack after VNF demand |

$avail_k = \min_{t \in [T_s, T_e]} (cap_k - used_k(t))$ over the request window.

### 4.2 Adjacency Matrix $A$

$$A_{ij} = \begin{cases}
1.0 & i = j \\
\dfrac{1}{delay(i,j) + 1} & \text{if a BW-feasible path exists between DC}_i \text{ and DC}_j \\
0 & \text{otherwise}
\end{cases}$$

The graph used to compute $delay(i,j)$ prunes all links with available bandwidth $< bw_{req}$, then runs `nx.shortest_path_length` with `weight="delay"`. Result is cached by `(T_{start}, T_{end}, round(bw, 1))`.

### 4.3 Normalized Adjacency

$$\hat{A} = D^{-1/2}(A + I)D^{-1/2}$$

where $D$ is the degree matrix of $A + I$. Cached by `A.tobytes()` (up to 256 entries).

### 4.4 GCN Encoder

Two GCN layers with shared-weight message passing:

$$H^{(1)} = \text{ReLU}(\hat{A} X W_1)$$
$$\mu = \hat{A} H^{(1)} W_\mu, \quad \log\sigma^2 = \text{clip}(\hat{A} H^{(1)} W_{lv},\ -10,\ 10)$$

**Training sample:** $Z = \mu + \sigma \odot \epsilon,\ \epsilon \sim \mathcal{N}(0,I)$

**Inference (deterministic):** $Z = \mu$

### 4.5 VGAE Loss

$$\mathcal{L}_{VGAE} = \mathcal{L}_{BCE} + \beta \cdot \mathcal{L}_{KL}$$

$$\mathcal{L}_{BCE} = -\mathbb{E}\left[A \log \sigma(ZZ^T) + (1-A)\log(1 - \sigma(ZZ^T))\right]$$

$$\mathcal{L}_{KL} = -\frac{1}{2}\mathbb{E}\left[1 + \log\sigma^2 - \mu^2 - \sigma^2\right]$$

An auxiliary reconstruction head penalizes deviation from original node features:
$$\mathcal{L}_{aux} = \|g(Z) - X\|^2_F, \quad g: \mathbb{R}^{d} \to \mathbb{R}^6$$

The combined loss uses an adaptive weight $\lambda$:
$$\mathcal{L} = \mathcal{L}_{VGAE} + \lambda \cdot \mathcal{L}_{aux}, \quad \lambda = \lambda_0 \cdot \frac{\mathcal{L}_{VGAE}}{\mathcal{L}_{VGAE} + \mathcal{L}_{aux} + \epsilon}$$

---

## 5. Placer DQN (`models/placer.py`)

### 5.1 State Vector

For each VNF placement step, the state is constructed as:

$$s = \left[\underbrace{\bar{Z}}_{\text{global context}} \;\Big|\; \underbrace{r_f}_{\text{VNF demand}} \;\Big|\; \underbrace{z_{prev}}_{\text{prev DC embedding}} \;\Big|\; \underbrace{p_{node}}_{\text{pressure}}\right] \in \mathbb{R}^{2 \cdot LATENT\_DIM + 4}$$

| Component | Dimension | Description |
|---|---|---|
| $\bar{Z} = \text{mean}(Z, \text{axis}=0)$ | `LATENT_DIM` | Global DC graph summary |
| $r_f = [mem_f, cpu_f, ram_f]$ | 3 | VNF resource demand |
| $z_{prev}$ | `LATENT_DIM` | Latent vector of the previously chosen DC (zeros for first VNF) |
| $p_{node}$ | 1 | Mean pressure over candidate DCs |

**Node pressure** for a single DC:
$$p = \frac{1}{|K|}\sum_{k \in K} \exp\!\left(-\frac{\max(avail_k - demand_k,\ 0)}{cap_k}\right)$$

Returns 1.0 immediately if any resource has no slack.

### 5.2 Network Architecture

Both policy and target networks are 3-layer MLPs:

```
Input(feat_dim) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(MAX_DCS)
```

An auxiliary **weight network** (same architecture, output dim=2) predicts per-request reward weights $(\alpha, \beta)$:
```
Input(feat_dim) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(2) → Sigmoid × [2.0, 1.0]
```

### 5.3 Action Selection

```
Q = policy_net(s)
mask invalid indices to -1e9
action = argmax(Q)          # greedy
        or random.choice()  # with probability epsilon
```

Valid indices: DC indices that (1) are within `MAX_DCS`, (2) the DC is in the VNF's affinity set, and (3) have sufficient resources in the request window.

### 5.4 Reward Function

$$R_{placer} = HRL\_R\_BASE\_LL + \alpha(1 - delay\_norm) - \beta \cdot cost\_norm - p_{node} - p_{path}$$

| Term | Formula | Description |
|---|---|---|
| $delay\_norm$ | $1 - \min\!\left(1, \frac{end\_time - t}{delay\_max}\right)$ | Urgency: 0 = much time left, 1 = at deadline |
| $cost\_norm$ | $\min\!\left(1, \frac{cost(vnf, dc)}{max\_cost}\right)$ | Normalized deployment cost |
| $p_{node}$ | PressureNode over chosen DC | DC congestion |
| $p_{path}$ | Mean link pressure over all SFC links | Network congestion |
| $\alpha, \beta$ | Output of weight network | Learned per-step trade-off |

**Failure penalty:** $R = -HRL\_PENALTY\_DROP = -0.5$

**Fallback (BestFit used):** $R = -HRL\_PENALTY\_DROP \times 0.5 = -0.25$

### 5.5 TD Training

Rewards are normalized by running mean and variance before computing TD targets:

$$\hat{r} = \frac{r - \mu_r}{\sigma_r + \epsilon}$$

$$\text{Target} = \hat{r} + \gamma \cdot \max_{a'} Q_{target}(s', a') \cdot (1 - done)$$

$$\mathcal{L}_{DQN} = \mathbb{E}\left[(Target - Q_{policy}(s, a))^2\right]$$

Weight network loss (supervised on positive rewards):
$$\mathcal{L}_{W} = \mathbb{E}\left[\left(\sigma(W(s)) - \frac{\max(r, 0)}{\max_r + \epsilon}\right)^2\right]$$

---

## 6. Routing (`strategy/routing_utils.py`)

### 6.1 Bandwidth-Pruned Graph

For each `(t_start, t_end, round(bw, 2))`, a pruned graph is built:
- Nodes: all network nodes
- Edges: only links with `available_bw(t_start, t_end) >= bw_req`

Edge weight used for shortest path:
$$w_{uv} = \frac{delay_{uv}}{max\_delay} + ROUTING\_PRESSURE\_WEIGHT \cdot \exp\!\left(-\frac{avail\_bw - bw_{req}}{cap}\right)$$

Constants: `ROUTING_BW_WEIGHT = 0.3`, `ROUTING_PRESSURE_WEIGHT = 0.5`.

### 6.2 Path Selection

`nx.shortest_path(G, u, v, weight="weight")` — cached by `(u, v, t_start, t_end, round(bw, 2))`.

If $u = v$, return `[u]` immediately (no routing needed).

### 6.3 Delay Check (in `DRL_Strategy.get_placement`)

After finding a path, delay is explicitly validated:
$$\sum_{l \in path} delay_l \leq delay\_max$$

If violated, the placement attempt returns `None`.

### 6.4 Path Pressure

Used in reward computation:
$$p_{path} = \frac{1}{|L|}\sum_{l \in L} pressure_l, \quad pressure_l = \begin{cases} 1.0 & avail_l < bw_{req} \\ \exp(-\frac{avail_l - bw_{req}}{cap_l}) & \text{otherwise} \end{cases}$$

---

## 7. Training Pipeline (`strategy/drl_strategy.py`, `utils/train.py`)

### 7.1 Pretraining Stage 1 — VGAE (`models/pretrain.py`)

1. For each training file, iterate all requests and build DC graph snapshots $(X, A)$.
2. Push snapshots into a `ReplayBuffer` (capacity 2000).
3. Train VGAE for `vgae_epochs` (default 60) epochs, batch size 16.
4. Save weights to `models/vgae_pretrained/vgae_weights.npy`.

### 7.2 Pretraining Stage 2 — Placer Behavioral Cloning (`models/pretrain.py`)

Uses `BestFit` as teacher:

1. For each episode, run `BestFit.get_placement` on every request.
2. If BestFit succeeds: extract per-VNF DC assignments as supervision signal.
3. If BestFit fails: generate random valid DC assignments as negative examples.
4. Compute reward using the weight network's current $(\alpha, \beta)$ output.
5. Push transition `(Z, vnf_feat, loc_z, pressure, action, reward, Z_next, valid_mask, ...)` to buffer.
6. Train placer every episode when buffer $\geq$ batch size.

Epsilon decays from 0.5 to 0.05 over episodes.

### 7.3 Online Training

**Multi-file training** (`utils/train.py`): episodes are distributed across training files. Weights are transferred between files; replay buffers are also carried over.

**Per-episode loop** (`DRL_Strategy.train`):

```
for ep in 1..episodes:
    reset env, clear caches
    pending = sort requests by arrival_time

    while pending:
        t = earliest arrival time
        batch = all requests arriving at t, filtered end_time > t
        sort batch by Earliest Deadline First (EDF)

        for sfc in batch:
            epsilon = cosine_annealing(progress, warmup=0.1)
            Z_t, dc_mapping = _get_z(t_start, t_end, bw, vnf_demand)

            with probability max(0.05, 0.3×(1−progress)):
                plan = BestFit (greedy teacher)
                rebuild trajectory from plan for buffer
            else:
                plan = get_placement(..., epsilon)   # DQN

            snap = snapshot_network()
            success, rewards = execute_with_fallback(plan, sfc, t, snap)

            if success:
                Z_next = _get_z(...)    # post-placement graph
                R = DRL_R_BASE_LL + α(1−time_ratio) − β·cost_norm
            else:
                R = −DRL_PENALTY_DROP
                restore_network(snap)

            push trajectory steps to buf_placer
            push (X, A) to buf_graph

            if total_steps % 4 == 0 and buf_placer ≥ DRL_BATCH_SIZE:
                placer.train(buf_placer)

            if total_steps % DRL_TARGET_SYNC == 0:
                placer.update_target_network()

            if total_steps % DRL_VGAE_TRAIN_FREQ == 0:
                vgae.train(buf_graph, epochs=DRL_VGAE_EPOCHS)
```

### 7.4 Epsilon Schedule

$$\epsilon(progress) = \begin{cases}
\epsilon_{max} & progress < warmup \\
\epsilon_{min} + \frac{\epsilon_{max} - \epsilon_{min}}{2}\left(1 + \cos(\pi \cdot t')\right) & \text{otherwise}
\end{cases}$$

where $t' = \frac{progress - warmup}{1 - warmup}$. Constants: $\epsilon_{max}=0.9,\ \epsilon_{min}=0.1,\ warmup=0.1$.

---

## 8. Evaluation (`strategy/drl_strategy.py:run_simulation_eval`)

1. Load model weights from directory (placer policy + weight net + VGAE).
2. Process requests deterministically ($\epsilon = 0$).
3. For each failed placement: attempt `BestFit` fallback.
4. Metrics reported:

| Metric | Formula |
|---|---|
| Acceptance Ratio | $accepted / total$ |
| Total Cost | $\sum_{accepted} \sum_{vnf} cost(vnf, dc)$ |
| Average Cost | $total\_cost / accepted$ |
| Total Delay | $\sum_{accepted} (end\_time - arrival\_time)$ |

---

## 9. Baseline Strategies (`strategy/`)

| Strategy | Description |
|---|---|
| `GreedyFIFS` | First-In First-Served; place VNFs on first feasible DC found |
| `BestFit` | Minimize resource waste: sort DCs by $\sum_k (avail_k - demand_k)$, pick smallest |
| `DeadlineAwareGreedy` | Prioritize requests with nearest deadline; use BestFit placement |
| `RandomFit` | Randomly select from feasible DCs |

All baselines use `RoutingMixin` for routing (same bandwidth-pruned shortest path logic).

---

## 10. Key Configuration (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `LATENT_DIM` | 8 | VGAE latent dimension |
| `MAX_DCS` | 60 | Max DC action space size |
| `TIMESTEP` | 0.1 | Time quantization unit |
| `DRL_BATCH_SIZE` | 32 | Placer training batch size |
| `DRL_TARGET_SYNC` | 40 | Steps between target network sync |
| `DRL_VGAE_TRAIN_FREQ` | 500 | Steps between online VGAE updates |
| `DRL_VGAE_EPOCHS` | 3 | Epochs per online VGAE update |
| `DRL_R_BASE_LL` | 1.0 | Base reward for successful placement |
| `DRL_PENALTY_DROP` | 0.5 | Penalty for failed placement |
| `DRL_LL_ALPHA` | 0.5 | Delay weight in reward |
| `DRL_LL_BETA` | 1.0 | Cost weight in reward |
| `EPSILON_MAX / MIN` | 0.9 / 0.1 | Exploration range |
| `EPSILON_WARMUP` | 0.1 | Warmup fraction before decay |
| `ROUTING_PRESSURE_WEIGHT` | 0.5 | BW pressure weight in routing cost |
| `DRL_MAX_GRAPH_CACHE` | 500 | LRU cache size for VGAE embeddings |