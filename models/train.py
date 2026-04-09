import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from env.vnf import VNF, ListOfVnfs
from env.request import Request, ListOfRequests
from env.network import Network
from env.env import Env
from strategy.hrl import HRL_VGAE_Strategy


def load_env(path: str) -> Env:
    with open(path) as f:
        data = json.load(f)
    network  = Network()
    vnfs     = ListOfVnfs()
    requests = ListOfRequests()

    for nid, nd in data.get("V", {}).items():
        if nd.get("server", False):
            network.add_dc_node(
                name=nid, delay=nd.get("d_v", 0.0),
                capacity={"mem": nd.get("h_v", 1.), "cpu": nd.get("c_v", 1.), "ram": nd.get("r_v", 1.)},
                cost={"mem": nd.get("cost_h", 1.), "cpu": nd.get("cost_c", 1.), "ram": nd.get("cost_r", 1.)})
        else:
            network.add_switch_node(nid)

    for lnk in data.get("E", []):
        network.add_link(str(lnk["u"]), str(lnk["v"]),
                         lnk.get("b_l", 1.0), lnk.get("d_l", 1.0))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f", 1.), c_f=vd.get("c_f", 1.), r_f=vd.get("r_f", 1.),
                         d_f={k: v for k, v in vd.get("d_f", {}).items()}))

    for idx, rd in enumerate(data.get("R", [])):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T", 0),
            delay_max=rd.get("d_max", 100.),
            start_node=str(rd.get("st_r", "")), end_node=str(rd.get("d_r", "")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r", [])],
            bandwidth=rd.get("b_r", 1.)))
    return Env(network, vnfs, requests)


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
        env = load_env(fp)
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

def main():
    ap = argparse.ArgumentParser(description="Train HRL-VGAE")
    ap.add_argument("--train-dir",     default="data/train")
    ap.add_argument("--episodes",      type=int, default=300)
    ap.add_argument("--ll-pretrained", default="models/ll_pretrained/ll_dqn_weights.weights.h5")
    ap.add_argument("--save-dir",      default="models/hrl_final")
    args = ap.parse_args()

    if not os.path.exists(args.ll_pretrained):
        args.ll_pretrained = None

    train(args.train_dir, args.episodes, args.ll_pretrained, args.save_dir)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()