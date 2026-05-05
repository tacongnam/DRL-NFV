import os
from data.load_data import get_data_files, print_selected_files, load_env_from_json

def _run_train(episodes, ll_pretrained, save_dir, train_dir, train_request_pct):
    files = get_data_files(train_dir)
    if not files:
        print(f"[ERROR] No training files in {train_dir}.")
        return None

    print_selected_files("TRAIN", files, request_pct=train_request_pct)

    n_files      = len(files)
    min_ep       = max(1, episodes // n_files)
    extra        = episodes % n_files
    total_ep_actual = min_ep * n_files + extra

    print(f"[TRAIN] {episodes} episodes across {n_files} files "
          f"(~{min_ep} ep/file, {extra} file(s) get +1) → total={total_ep_actual}")

    strategy = None
    for i, fp in enumerate(files):
        ep_for_file = min_ep + (1 if i < extra else 0)

        print(f"\n--- File {i+1}/{n_files}: {os.path.basename(fp)} ({ep_for_file} ep) ---")
        env = load_env_from_json(fp, request_pct=train_request_pct)
        from strategy import HRL_VGAE_Strategy
        strategy = HRL_VGAE_Strategy(
            env, is_training=True, episodes=ep_for_file,
            use_ll_score=True,
            ll_pretrained_path=ll_pretrained if i == 0 else None)

        if i > 0:
            hl_w = os.path.join(save_dir, "hl_pmdrl_weights.npy")
            ll_w = os.path.join(save_dir, "ll_dqn_weights.npy")
            if os.path.exists(hl_w) or os.path.exists(ll_w):
                strategy.load_model(save_dir)

        env.set_strategy(strategy)
        env.run_simulation()
        os.makedirs(save_dir, exist_ok=True)
        strategy.save_model(save_dir)

    if strategy:
        strategy.save_model(save_dir)
    return strategy