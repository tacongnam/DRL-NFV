# DRL-NFV: Deep Reinforcement Learning for NFV Placement

---

## 1. Problem Definition

**Input:**
- Physical network $G = (N, L)$ with nodes $N$ (Data Centers and Switches) and links $L$
- VNF catalog: each function $f$ has resource demand $r_f = \{mem_f, cpu_f, ram_f\}$ and optional DC affinity
- Request set $R$: each request $req_k$ carries $(arrival\_time, delay\_max, start\_node, end\_node, [f_1,\ldots,f_m], bw)$
- $end\_time_k = arrival\_time_k + delay\_max_k$

**Constraints:**
- Node capacity: $\sum_{f \in placed(n,t)} r_f[k] \leq cap_n[k]$ for every active timeslot and resource type
- Link bandwidth: $used\_bw_{l,t} + bw_{req} \leq cap_l$ for every active timeslot
- Path delay: $\sum_{l \in path} delay_l \leq delay\_max$
- VNF ordering: VNFs in each SFC must be chained in sequence through selected DCs and the final destination

**Timeslot conversion:**
$$T = \left\lfloor \frac{t}{TIMESTEP} + 0.5 \right\rfloor, \quad TIMESTEP = 0.1$$

---

## 2. System Overview

The active algorithm implemented in this repository is `DRL-NFV`. It combines:

- a VGAE encoder for DC graph and network state,
- a DQN-based placer agent for VNF-to-DC assignment,
- online graph buffering and periodic VGAE fine-tuning.

The main learned decision policy is the placer DQN; the VGAE provides latent state embeddings for current request windows.

---

## 3. VGAE Graph Encoding

### 3.1 DC Node Features

Each data center node is represented using normalized available resources and resource slack after the request demand.

### 3.2 Adjacency

Two DCs are connected when a feasible bandwidth-pruned path exists between them. The adjacency value is:
$$A_{ij} = \begin{cases}
1.0 & i = j \\
\frac{1}{delay(i,j) + 1} & \text{if a path exists for requested bandwidth} \\
0 & \text{otherwise}
\end{cases}$$

### 3.3 VGAE Output

The VGAE encodes the graph into a latent matrix $Z \in \mathbb{R}^{|DC| \times LATENT\_DIM}$. During inference the encoder uses deterministic latent means.

---

## 4. Placer DQN

### 4.1 State Representation

For each VNF placement step, the placer state is built from:
- a global summary of $Z$,
- the VNF resource demand vector,
- the latent embedding of the previously selected DC,
- a scalar node pressure feature.

This state vector is used as input to both the placer policy network and the auxiliary weight network.

### 4.2 Action Space

The action selects a DC index among valid candidate centers. The policy outputs Q-values for `MAX_DCS` slots and invalid actions are masked before argmax.

### 4.3 Reward

The placer reward balances delay, cost, and resource pressure:
$$R = HRL\_R\_BASE\_LL + \alpha (1 - delay\_norm) - \beta cost\_norm - pressure - path\_pressure$$

Where:
- $delay\_norm$ measures closeness to the request deadline,
- $cost\_norm$ is the normalized deployment cost for the VNF,
- $pressure$ captures DC resource congestion,
- $path\_pressure$ captures routing pressure.

A failed placement receives a negative drop penalty.

### 4.4 Training

The placer DQN uses experience replay and target network updates. Rewards are normalized by a running mean and variance before computing TD targets.

---

## 5. DRL-NFV Training Pipeline

### 5.1 Pretraining

The pipeline supports two pretraining stages:

- VGAE pretraining on DC graph snapshots from training requests,
- placer pretraining using `BestFit` as a behavioral cloning teacher.

### 5.2 Online Training

Each training episode follows these steps:
1. Reset the environment and sort pending requests by arrival time.
2. Collect requests that are active at the current timeslot.
3. Sort active requests by earliest deadline first.
4. For each request:
   - compute the DC graph embedding $Z_t$ for the request window,
   - build valid DC candidate lists for each VNF,
   - select DCs with the placer DQN using epsilon-greedy exploration,
   - route the service chain through the network,
   - if placement fails, attempt `BestFit` fallback.

### 5.3 Greedy Fallback

A small fraction of training steps uses `BestFit` as a greedy teacher override to stabilize learning. Failed placements are penalized and the environment state is restored when necessary.

### 5.4 Online Updates

Online updates include:
- placer training every 4 placement steps when the replay buffer is ready,
- target network sync every `HRL_TARGET_SYNC` steps,
- VGAE fine-tuning every `HRL_VGAE_TRAIN_FREQ` steps from graph snapshots.

---

## 6. Routing

Routing uses `RoutingMixin` with bandwidth-pruned shortest paths. A path is feasible when:
- every link on the path has enough available bandwidth,
- the total delay does not exceed the request deadline.

Path caches are cleared after successful placement steps.

---

## 7. Evaluation

Evaluation runs the trained `DRL-NFV` model deterministically (`epsilon=0.0`) and uses `BestFit` fallback for failed placement attempts. Reported metrics include acceptance ratio, total cost, average cost, and total delay.

---

## 8. Legacy Note

Many configuration keys still use the legacy `HRL_*` prefix, but the active strategy in this repository is implemented in `strategy/drl_strategy.py` as `DRL_Strategy`.

