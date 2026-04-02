import os
import math
import copy
import numpy as np
import networkx as nx
from typing import Dict, List, Optional

import config
from env.env import Strategy
from env.request import SFC
from models.model import ReplayBuffer, VGAENetwork, HighLevelAgent, LowLevelAgent


class HRL_VGAE_Strategy(Strategy):
    def __init__(self, env, is_training: bool = False, episodes: int = 300,
                 use_ll_score: bool = True, ll_pretrained_path: str = None):
        super().__init__(env)
        self.name = "HRL-VGAE-Simplified"
        self.is_training = is_training
        self.episodes = episodes

        latent_dim = 8
        ll_input_dim = latent_dim + 3
        hl_input_dim = latent_dim + (4 if use_ll_score else 3)

        self.vgae_net = VGAENetwork()
        self.hl_agent = HighLevelAgent(gamma=0.95, use_ll_score=use_ll_score, input_dim=hl_input_dim)
        self.ll_agent = LowLevelAgent(gamma=0.95, input_dim=ll_input_dim)

        self.buffer_HL = ReplayBuffer(capacity=5000)
        self.buffer_LL = ReplayBuffer(capacity=10000)
        self.buffer_Graph = ReplayBuffer(capacity=2000)

        self.last_ll_trajectory = []
        self.use_ll_score = use_ll_score
        self.ll_pretrained_path = ll_pretrained_path

        if ll_pretrained_path:
            self._load_pretrained_ll(ll_pretrained_path)

    def _load_pretrained_ll(self, path: str):
        try:
            # Check if path is a directory or file
            if os.path.isdir(path):
                # If directory, look for ll_dqn_weights.weights.h5
                weight_file = os.path.join(path, "ll_dqn_weights.weights.h5")
            elif path.endswith('.weights.h5'):
                # If already has extension, use as-is
                weight_file = path
            else:
                # If file without extension, add .weights.h5
                weight_file = path + '.weights.h5'
            
            if os.path.exists(weight_file):
                self.ll_agent.policy_net.load_weights(weight_file)
                print(f"[HRL] Loaded pre-trained LL model from {weight_file}")
            else:
                print(f"[HRL] Warning: Pre-trained LL model not found at {weight_file}")
        except Exception as e:
            print(f"[HRL] Warning: Could not load pretrained LL model: {e}")

    def save_model(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        ll_path = os.path.join(directory, "ll_dqn_weights.weights.h5")
        hl_path = os.path.join(directory, "hl_pmdrl_weights.weights.h5")
        try:
            self.ll_agent.policy_net.save_weights(ll_path)
            self.hl_agent.policy_net.save_weights(hl_path)
            print(f"[HRL] Models saved to {directory}")
        except Exception as e:
            print(f"[HRL] Warning: Could not save model: {e}")

    def load_model(self, directory: str):
        import os
        ll_path = os.path.join(directory, "ll_dqn_weights.weights.h5")
        hl_path = os.path.join(directory, "hl_pmdrl_weights.weights.h5")

        # Policy net is already built, load weights directly
        if os.path.exists(ll_path):
            try:
                self.ll_agent.policy_net.load_weights(ll_path)
                print(f"[HRL] Loaded LL model from {ll_path}")
            except Exception as e:
                print(f"[HRL] Warning: Could not load LL model: {e}")

        if os.path.exists(hl_path):
            try:
                self.hl_agent.policy_net.load_weights(hl_path)
                print(f"[HRL] Loaded HL model from {hl_path}")
            except Exception as e:
                print(f"[HRL] Warning: Could not load HL model: {e}")

    def get_routing(self, u: str, v: str, t_start: int, t_end: int, bw: float) -> Optional[List[str]]:
        u, v = str(u), str(v)
        if u == v:
            return [u]

        G = self.env.network.to_graph()
        if u not in G or v not in G:
            return None

        edges_to_remove = []
        for a, b in G.edges():
            link_obj = next((l for l in self.env.network.links if {l.u.name, l.v.name} == {a, b}), None)
            if not link_obj or link_obj.get_available_bandwidth(t_start, t_end) < bw:
                edges_to_remove.append((a, b))

        G.remove_edges_from(edges_to_remove)

        try:
            return nx.shortest_path(G, u, v, weight='delay')
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None

    def build_dc_full_graph(self, t_start: int, t_end: int, bw_req: float = 0.0):
        dcs = [nid for nid, n in self.env.network.nodes.items() if n.type == config.NODE_DC]
        n = len(dcs)
        X = []
        A = np.zeros((n, n))

        for dc_id in dcs:
            res = self.env.network.nodes[dc_id].get_min_available_resource(t_start, t_end)
            X.append([res["mem"], res["cpu"], res["ram"]])

        for i in range(n):
            for j in range(n):
                if i != j:
                    path = self.get_routing(dcs[i], dcs[j], t_start, t_end, bw_req)
                    A[i][j] = 1.0 / len(path) if path else 0.0

        return np.array(X, dtype=np.float32), A, dcs

    def get_placement(self, sfc: SFC, current_time: float, Z_t=None, dc_mapping=None, epsilon: float = 0.0) -> Optional[Dict]:
        self.last_ll_trajectory = []
        t_start = self.env._get_timeslot(current_time)
        t_end = self.env._get_timeslot(sfc.request.end_time)

        node_placements, vnf_timeslots = [], []
        link_paths, link_timeslots = [], []
        prev_dc = sfc.request.start_node

        if Z_t is None or dc_mapping is None:
            X, A, dc_mapping = self.build_dc_full_graph(t_start, t_end, sfc.request.bw)
            Z_t = self.vgae_net.encode(X, A)

        for vnf in sfc.request.vnfs:
            candidate_dcs = [str(x) for x in vnf.get_dcs()]
            if '-1' in candidate_dcs or not candidate_dcs:
                candidate_dcs = dc_mapping

            valid_indices = [
                idx for idx, dc_id in enumerate(dc_mapping)
                if dc_id in candidate_dcs
                and self.env.network.nodes[dc_id].type == config.NODE_DC
                and self.env._check_can_deploy_vnf(self.env.network.nodes[dc_id], vnf, t_start, t_end)
            ]

            if not valid_indices:
                return None

            vnf_req = [vnf.resource["mem"], vnf.resource["cpu"], vnf.resource["ram"]]
            action_idx = self.ll_agent.act(Z_t, vnf_req, valid_indices, epsilon)
            chosen_dc = dc_mapping[action_idx]

            self.last_ll_trajectory.append({
                'Z_t': Z_t,
                'vnf_feat': vnf_req,
                'action_idx': action_idx,
                'valid_mask': valid_indices,
            })

            path = self.get_routing(prev_dc, chosen_dc, t_start, t_end, sfc.request.bw)
            if path is None:
                return None

            node_placements.append(chosen_dc)
            vnf_timeslots.append((t_start, t_end))
            link_paths.append(path)
            link_timeslots.append((t_start, t_end))
            prev_dc = chosen_dc

        final_path = self.get_routing(prev_dc, sfc.request.end_node, t_start, t_end, sfc.request.bw)
        if final_path is None:
            return None

        link_paths.append(final_path)
        link_timeslots.append((t_start, t_end))

        return self.build_placement_plan(node_placements, link_paths, vnf_timeslots, link_timeslots, sfc)

    def train(self) -> dict:
        import time
        BATCH_SIZE = 32
        total_steps = 0
        max_total_steps = self.episodes * 20  # Target for epsilon ~0.01
        start_time = time.time()

        # Phase 1: Warm-up with greedy placement (first 30% of episodes)
        warmup_episodes = max(1, int(self.episodes * 0.3))

        # Progress tracking variables
        episode_times = []
        best_acc_rate = 0.0

        for episode in range(1, self.episodes + 1):
            episode_start = time.time()
            self.env.reset()
            waitlist = sorted([SFC(r) for r in self.env.requests], key=lambda s: s.request.arrival_time)
            queue = []
            t_step = 0
            done = False
            ep_accepted = 0
            ep_rejected = 0
            
            # Initialize epsilon before the loop to avoid UnboundLocalError
            if episode <= warmup_episodes:
                epsilon = 0.0
            else:
                progress = (episode - warmup_episodes) / (self.episodes - warmup_episodes)
                epsilon = 0.01 + 0.3 * math.exp(-8 * progress)

            while not done:
                self.env.t = t_step * config.TIMESTEP

                while waitlist and waitlist[0].request.arrival_time <= self.env.t:
                    queue.append(waitlist.pop(0))

                queue = [s for s in queue if self.env.t <= s.request.end_time]

                if not queue and not waitlist:
                    done = True
                    break

                total_steps += 1

                # Epsilon: warmup phase uses 0 (greedy), then gradual exploration
                if episode <= warmup_episodes:
                    epsilon = 0.0
                else:
                    progress = (episode - warmup_episodes) / (self.episodes - warmup_episodes)
                    epsilon = 0.01 + 0.3 * math.exp(-8 * progress)

                t_end_eval = int(self.env.t / config.TIMESTEP) + 10
                bw_req = queue[0].request.bw if queue else 0.0
                X, A, dc_mapping = self.build_dc_full_graph(int(self.env.t / config.TIMESTEP), t_end_eval, bw_req)
                Z_t = self.vgae_net.encode(X, A)

                if queue:
                    Z_mean_t = np.mean(Z_t, axis=0, keepdims=True)
                    sfc_feats_t = self.hl_agent.extract_sfc_features(
                        queue,
                        Z_t=Z_t if self.use_ll_score else None,
                        ll_agent=self.ll_agent if self.use_ll_score else None
                    )

                    # LL uses higher epsilon in warmup to encourage exploration
                    if episode <= warmup_episodes:
                        ll_epsilon = 0.3  # Higher exploration in warmup
                    else:
                        ll_epsilon = 0.05  # Almost greedy after warmup

                    sfc_idx = self.hl_agent.act(Z_t, queue, epsilon, ll_agent=self.ll_agent)
                    selected_sfc = queue.pop(sfc_idx)

                    S_backup = copy.deepcopy(self.env.network)
                    plan = self.get_placement(selected_sfc, self.env.t, Z_t, dc_mapping, ll_epsilon)
                    success, rewards, _ = self.env.step(plan)

                    if success:
                        ep_accepted += 1
                        self.env.stats['accepted_requests'] += 1
                        self.env.stats['total_cost'] += -rewards[1] if len(rewards) > 1 else 0
                        time_rem = selected_sfc.request.end_time - self.env.t
                        tMax = selected_sfc.request.delay_max
                        # Increased reward for acceptance to encourage learning
                        R_HL = [2.0, rewards[1] if len(rewards) > 1 else 0]
                        R_LL = 2.0 + 0.5 * max(0.0, time_rem / tMax) + (rewards[1] if len(rewards) > 1 else 0)
                    else:
                        ep_rejected += 1
                        self.env.stats['rejected_requests'] += 1
                        self.env.network = S_backup
                        # Reduced penalty for failure to avoid discouraging exploration
                        R_HL = [-0.1, 0.0]
                        R_LL = -0.1
                        if self.env.t < selected_sfc.request.end_time:
                            queue.append(selected_sfc)

                    X_next, A_next, _ = self.build_dc_full_graph(int(self.env.t / config.TIMESTEP), t_end_eval, bw_req)
                    Z_next = self.vgae_net.encode(X_next, A_next)
                    Z_mean_next = np.mean(Z_next, axis=0, keepdims=True)
                    sfc_feats_next = self.hl_agent.extract_sfc_features(
                        queue,
                        Z_t=Z_next if self.use_ll_score else None,
                        ll_agent=self.ll_agent if self.use_ll_score else None
                    )

                    self.buffer_HL.push((Z_mean_t, sfc_feats_t, sfc_idx, R_HL, Z_mean_next, sfc_feats_next, done))

                    for i, step_log in enumerate(self.last_ll_trajectory):
                        nxt_mask = self.last_ll_trajectory[i + 1]['valid_mask'] if i + 1 < len(self.last_ll_trajectory) else []
                        self.buffer_LL.push((
                            step_log['Z_t'],
                            np.array([step_log['vnf_feat']], dtype=np.float32),
                            step_log['action_idx'],
                            R_LL,
                            Z_next, nxt_mask, done
                        ))

                if total_steps % 50 == 0:
                    self.buffer_Graph.push((X, A))

                # Train only when we have enough samples
                if total_steps >= BATCH_SIZE and len(self.buffer_Graph) > BATCH_SIZE:
                    self.vgae_net.train(self.buffer_Graph, epochs=1)
                if len(self.buffer_LL) > BATCH_SIZE:
                    self.ll_agent.train(self.buffer_LL, BATCH_SIZE)
                if len(self.buffer_HL) > BATCH_SIZE:
                    self.hl_agent.train(self.buffer_HL, BATCH_SIZE)

                if total_steps % 200 == 0:
                    self.ll_agent.update_target_network()
                    self.hl_agent.update_target_network()

                t_step += 1

            # Episode completed - calculate metrics
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            acc_rate = ep_accepted / max(1, ep_accepted + ep_rejected)
            
            # Track best accuracy
            if acc_rate > best_acc_rate:
                best_acc_rate = acc_rate
            
            # Calculate progress and ETA
            elapsed_total = time.time() - start_time
            progress_pct = episode / self.episodes * 100
            avg_episode_time = np.mean(episode_times[-10:]) if episode_times else episode_time
            remaining_episodes = self.episodes - episode
            eta_seconds = remaining_episodes * avg_episode_time
            
            # Format ETA
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
            
            # Progress bar
            bar_length = 30
            filled = int(bar_length * episode / self.episodes)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Print detailed progress
            print(f"\r[{bar}] {progress_pct:5.1f}% | Ep {episode}/{self.episodes} | "
                  f"Acc: {ep_accepted}/{ep_accepted+ep_rejected} ({acc_rate:.1%}) | "
                  f"Best: {best_acc_rate:.1%} | ε: {epsilon:.3f} | "
                  f"ETA: {eta_str} | Elapsed: {elapsed_total:.0f}s", end="", flush=True)
            
            # New line every 10 episodes for readability
            if episode % 50 == 0:
                print()

        total_time = time.time() - start_time
        print(f"\n\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Total steps: {total_steps}")
        print(f"Best acceptance rate: {best_acc_rate:.1%}")
        
        acc_ratio = self.env.stats['accepted_requests'] / max(1, self.env.stats['accepted_requests'] + self.env.stats['rejected_requests'])
        print(f"Final acceptance ratio: {acc_ratio:.3f}")

        total_req = self.env.stats['total_requests'] * self.episodes
        self.env.stats['acceptance_ratio'] = self.env.stats['accepted_requests'] / total_req if total_req > 0 else 0.0
        self.env.stats['algorithm_name'] = self.name
        return self.env.stats
