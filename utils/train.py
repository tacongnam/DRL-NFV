import os
import numpy as np
from data.load_data import get_data_files, print_selected_files, load_env_from_json

def _run_train(episodes, ll_pretrained, save_dir, train_dir, train_request_pct, logger=None):
    files = get_data_files(train_dir)
    if not files:
        print(f"[ERROR] No training files in {train_dir}.")
        return None

    print_selected_files("TRAIN", files, request_pct=train_request_pct)

    n_files         = len(files)
    min_ep          = max(1, episodes // n_files)
    extra           = episodes % n_files
    total_ep_actual = min_ep * n_files + extra

    print(f"[TRAIN] {episodes} episodes across {n_files} files "
          f"(~{min_ep} ep/file, {extra} file(s) get +1) → total={total_ep_actual}")

    from strategy import HRL_VGAE_Strategy
    prev_strategy = None
    episode_offset = 0

    for i, fp in enumerate(files):
        ep_for_file = min_ep + (1 if i < extra else 0)
        print(f"\n--- File {i+1}/{n_files}: {os.path.basename(fp)} ({ep_for_file} ep) ---")
        env = load_env_from_json(fp, request_pct=train_request_pct)
        strategy = HRL_VGAE_Strategy(
            env, is_training=True, episodes=ep_for_file,
            use_ll_score=True,
            ll_pretrained_path=ll_pretrained if i == 0 else None,
            logger=logger,
            episode_offset=episode_offset)

        if prev_strategy is not None:
            dummy_X = np.zeros((2, 3), np.float32)
            dummy_A = np.eye(2, dtype=np.float32)
            strategy.vgae_net.encode(dummy_X, dummy_A)

            strategy.hl_agent.set_weights(prev_strategy.hl_agent.get_weights())
            strategy.ll_agent.policy_net.set_weights(prev_strategy.ll_agent.policy_net.get_weights())
            strategy.ll_agent.weight_net.set_weights(prev_strategy.ll_agent.weight_net.get_weights())
            strategy.vgae_net.gcn1.set_weights(prev_strategy.vgae_net.gcn1.get_weights())
            strategy.vgae_net.gcn_mu.set_weights(prev_strategy.vgae_net.gcn_mu.get_weights())
            strategy.vgae_net.gcn_lv.set_weights(prev_strategy.vgae_net.gcn_lv.get_weights())
            strategy.buf_HL = prev_strategy.buf_HL
            strategy.buf_LL = prev_strategy.buf_LL
            strategy.buf_Graph = prev_strategy.buf_Graph
        elif i > 0:
            hl_w = os.path.join(save_dir, "hl_pmdrl_weights.npy")
            ll_w = os.path.join(save_dir, "ll_dqn_weights.npy")
            if os.path.exists(hl_w) or os.path.exists(ll_w):
                strategy.load_model(save_dir)

        env.set_strategy(strategy)
        env.run_simulation()
        os.makedirs(save_dir, exist_ok=True)
        strategy.save_model(save_dir)

        prev_strategy  = strategy
        episode_offset += ep_for_file

    if prev_strategy:
        prev_strategy.save_model(save_dir)
    return prev_strategy