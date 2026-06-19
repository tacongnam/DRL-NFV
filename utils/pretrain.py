import os
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

def _run_pretrain_inline(args, train_dir: str, logger=None):
    from models import pretrain
    try:
        selected = sorted(os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.json'))
    except FileNotFoundError:
        selected = []

    if not selected:
        logger.warning("[Pretrain] No training files selected.")
        return False

    logger.info("[Pretrain] Running inline on %s", train_dir)

    vgae = pretrain.pretrain_vgae(selected, epochs=getattr(args, "vgae_epochs", 60), logger=logger)

    if getattr(args, "ll_episodes", 0) > 0:
        pretrain.pretrain_placer(
            selected,
            vgae,
            episodes=getattr(args, "ll_episodes", 60),
            logger=logger,
        )

    import config
    vgae_out = os.path.join(config.VGAE_DIR, config.VGAE_WEIGHTS_FILE)
    placer_out = os.path.join(config.PLACER_DIR, config.PLACER_WEIGHTS_FILE)
    logger.info("[Pretrain] VGAE saved: %s -> %s", os.path.exists(vgae_out), vgae_out)
    logger.info("[Pretrain] Placer saved: %s -> %s", os.path.exists(placer_out), placer_out)
    return os.path.exists(vgae_out) or os.path.exists(placer_out)