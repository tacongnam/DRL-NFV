import os
import numpy as np
from data.load_data import get_data_files, print_selected_files, load_env_from_json

def _run_train(episodes, ll_pretrained, save_dir, train_dir, train_request_pct, logger=None):
    files = get_data_files(train_dir)
    if not files:
        print(f"[ERROR] No training files in {train_dir}.")
        return None

    print_selected_files("TRAIN", files, request_pct=train_request_pct)

    import random
    all_tasks = []
    while len(all_tasks) < episodes:
        shuffled_files = list(files)
        random.shuffle(shuffled_files)
        all_tasks.extend(shuffled_files)
    all_tasks = all_tasks[:episodes]

    from strategy.drl_strategy import DRL_Strategy
    prev_strategy  = None
    episode_offset = 0

    for i, fp in enumerate(files):
        if logger:
            logger.mark_file_boundary(fp)

        env = load_env_from_json(fp, request_pct=train_request_pct)
        strategy = DRL_Strategy(
            env, is_training=True, episodes=1,
            placer_pretrained_path=ll_pretrained if i == 0 else None,
            logger=logger,
            episode_offset=episode_offset)

        if prev_strategy is not None:
            dummy_X = np.zeros((2, strategy.vgae_net.NODE_FEAT_DIM), np.float32)
            dummy_A = np.eye(2, dtype=np.float32)
            strategy.vgae_net.encode(dummy_X, dummy_A)

            strategy.placer.policy_net.set_weights(prev_strategy.placer.policy_net.get_weights())
            strategy.placer.weight_net.set_weights(prev_strategy.placer.weight_net.get_weights())
            strategy.vgae_net.gcn1.set_weights(prev_strategy.vgae_net.gcn1.get_weights())
            strategy.vgae_net.gcn_mu.set_weights(prev_strategy.vgae_net.gcn_mu.get_weights())
            strategy.vgae_net.gcn_lv.set_weights(prev_strategy.vgae_net.gcn_lv.get_weights())
            strategy.buf_placer = prev_strategy.buf_placer
            strategy.buf_graph  = prev_strategy.buf_graph
            strategy.vgae_net.freeze_backbone()

        env.set_strategy(strategy)
        env.run_simulation()

        if i % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            strategy.save_model(save_dir)

        prev_strategy   = strategy

    if prev_strategy:
        prev_strategy.save_model(save_dir)
    return prev_strategy