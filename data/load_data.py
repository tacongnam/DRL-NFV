from env import VNF, ListOfVnfs, Request, ListOfRequests, Network, Env
import json, os, random, csv
from utils.helpers import sample_requests, resolve_request_limit

def load_env_from_json(filepath: str, request_pct: int = 0) -> Env:
    with open(filepath) as f:
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
                         lnk.get("b_l", 1.), lnk.get("d_l", 1.))

    for idx, vd in enumerate(data.get("F", [])):
        vnfs.add_vnf(VNF(idx,
                         h_f=vd.get("h_f", 1.), c_f=vd.get("c_f", 1.), r_f=vd.get("r_f", 1.),
                         d_f={k: v for k, v in vd.get("d_f", {}).items()}))

    req_rows = sorted(data.get("R", []), key=lambda r: r.get("T", 0))
    req_rows = sample_requests(req_rows, request_pct=request_pct)  # From utils.helpers

    for idx, rd in enumerate(req_rows):
        requests.add_request(Request(
            name=idx, arrival_time=rd.get("T", 0),
            delay_max=rd.get("d_max", 100.),
            start_node=str(rd.get("st_r", "")), end_node=str(rd.get("d_r", "")),
            VNFs=[vnfs.vnfs[str(vi)] for vi in rd.get("F_r", [])],
            bandwidth=rd.get("b_r", 1.)))
    return Env(network, vnfs, requests)


def get_data_files(d: str):
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    return []


def sample_files(files: list, n: int | None, seed: int | None = None) -> list:
    """Randomly sample n files from files list. Returns all if n is None or >= len."""
    if not files or n is None or n <= 0 or n >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, n))


def print_selected_files(label: str, files: list, request_pct: int = 0):
    print(f"\n[{label}] Selected {len(files)} file(s)")
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        total_requests = len(data.get("R", []))
        req_limit = resolve_request_limit(total_requests, request_pct=request_pct)
        req_label = total_requests if req_limit is None else min(total_requests, req_limit)
        print(f"  - {os.path.basename(fp)}: req={req_label}/{total_requests}")


def save_csv(results: list, path: str, fieldnames: list = None):
    if not results:
        return
    fieldnames = fieldnames or list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[CSV] Saved → {path}")