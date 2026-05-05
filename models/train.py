import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from strategy import HRL_VGAE_Strategy
from data.load_data import load_env_from_json

def get_files(d: str):
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    return []


def train(train_dir: str, episodes: int, ll_pretrained: str, save_dir: str):
    files = get_files(train_dir)
    if not files:
        print(f"No training files in {train_dir}")
        return None

    strategy = None
    for i, fp in enumerate(files):
        print(f"\n--- File {i+1}/{len(files)}: {os.path.basename(fp)} ---")
        env = load_env_from_json(fp)
        strategy = HRL_VGAE_Strategy(
            env,
            is_training=True,
            episodes=episodes,
            use_ll_score=True,
            ll_pretrained_path=ll_pretrained if i == 0 else None,
        )

        if i > 0 and strategy is not None:
            hl_w = os.path.join(save_dir, "hl_pmdrl_weights.weights.h5")
            if os.path.exists(hl_w):
                strategy.load_model(save_dir)

        env.set_strategy(strategy)
        env.run_simulation()

    os.makedirs(save_dir, exist_ok=True)
    if strategy:
        strategy.save_model(save_dir)
    return strategy