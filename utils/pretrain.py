import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _run_pretrain_inline(args, train_dir: str, DEFAULT_PRETRAIN_REQUEST_PCT):
    from models import pretrain

    selected = pretrain.get_train_files(train_dir)
    if not selected:
        print("[Pretrain] No training files selected.", flush=True)
        return False

    req_pct = getattr(args, "pretrain_request_pct", DEFAULT_PRETRAIN_REQUEST_PCT)
    pretrain.print_selected_files(selected, req_pct)

    print(f"[Pretrain] Running inline on {train_dir}", flush=True)
    vgae = None
    vgae = pretrain.pretrain_vgae(
        selected,
        epochs=getattr(args, "vgae_epochs", 60),
        request_pct=req_pct,
    )
    if vgae is None and getattr(args, "ll_episodes", 0) > 0:
        vgae_path = os.path.join(ROOT_DIR, "models", "vgae_pretrained", "vgae_weights.npy")
        if os.path.exists(vgae_path):
            vgae = pretrain.VGAENetwork(latent_dim=pretrain.LATENT_DIM)
            vgae.load_weights(vgae_path)

    if vgae is not None:
        pretrain.pretrain_ll(
            selected,
            vgae,
            episodes=getattr(args, "ll_episodes", 60),
            request_pct=req_pct,
        )
    else:
        print("[Pretrain] Skipped LL pretrain because VGAE was not produced.", flush=True)

    vgae_out = os.path.join(ROOT_DIR, "models", "vgae_pretrained", "vgae_weights.npy")
    ll_out = os.path.join(ROOT_DIR, "models", "ll_pretrained", "ll_dqn_weights.npy")
    print(f"[Pretrain] VGAE saved: {os.path.exists(vgae_out)} -> {vgae_out}", flush=True)
    print(f"[Pretrain] LL saved: {os.path.exists(ll_out)} -> {ll_out}", flush=True)
    return os.path.exists(vgae_out) or os.path.exists(ll_out)
