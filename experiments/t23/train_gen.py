"""
train_gen.py — train the Ali-AUG generator for ONE split of the Tables 2-3 re-run.

Runs the RELEASED trainer (ali_sd_public/train_ali2ali.py) unmodified, with two
harness-level changes that do not alter the method:
  * wandb is replaced by a no-op stub BEFORE the trainer is imported, because the
    released file calls wandb.login(key=...) with a hard-coded key at import time;
  * python/numpy/torch are seeded from --seed (the released trainer never seeds).

Recipe (decided 2026-09-26, see prep/t23/PLAN.md): no conditioning dropout (as in the
published results), 10,000 steps, lambda_lpips 10 (paper table; the released default
is 100), LoRA rank 8/4 (hard-coded in the released trainer), batch 1, lr 5e-4.

    python prep/t23/train_gen.py --code <ali_sd_public> --split <.../split_k> --out <dir>
"""
import argparse
import random
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="path to ali_sd_public")
ap.add_argument("--split", required=True, help="split dir with train_A/B/C + train_prompts.json")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=0,
                help="dataloader workers; keep 0 on Windows (spawn re-imports this script)")
ap.add_argument("--fix-sched", action="store_true",
                help="re-enable set_timesteps(1) (see fixsched.py); without it the U-Net is inert")
ap.add_argument("--no-skip", action="store_true", help="ablation: no encoder->decoder skips (ablations.py)")
ap.add_argument("--lambda-clip", type=float, default=5.0, help="0 = ablation without CLIP similarity loss")
ap.add_argument("--lambda-gan", type=float, default=2.5, help="0 = ablation without adversarial loss")
a = ap.parse_args()

# --- neutralise wandb (must precede the trainer import) ------------------------------
# Real wandb if installed (disabled mode, login made a no-op so the hard-coded key is
# never sent); otherwise a stub with a __spec__, since some libraries probe
# importlib.util.find_spec("wandb").
import importlib.machinery  # noqa: E402
import os  # noqa: E402

os.environ["WANDB_MODE"] = "disabled"
try:
    import wandb as wb
except ImportError:
    wb = types.ModuleType("wandb")
    wb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    wb.init = lambda *x, **k: None
    wb.log = lambda *x, **k: None
    wb.Image = lambda *x, **k: None
    sys.modules["wandb"] = wb
wb.login = lambda *x, **k: None
# even disabled, real wandb.Image writes temp files; two concurrent runs collided on them
wb.init = lambda *x, **k: None
wb.log = lambda *x, **k: None
wb.Image = lambda *x, **k: None

import numpy as np  # noqa: E402
import torch  # noqa: E402

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)

sys.path.insert(0, a.code)
if a.fix_sched:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixsched  # noqa: E402
    fixsched.apply()
if a.no_skip:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ali2ali  # noqa: E402,F401
    import ablations  # noqa: E402
    ablations.apply_no_skip()
import train_ali2ali  # noqa: E402
from utils import parse_args_paired_training  # noqa: E402

args = parse_args_paired_training([
    "--dataset_folder", a.split,
    "--output_dir", a.out,
    "--pretrained_model_name_or_path", "stabilityai/sd-turbo",
    "--train_batch_size", "1",
    "--max_train_steps", str(a.steps),
    "--checkpointing_steps", "2500",
    "--viz_freq", "100000",
    "--learning_rate", "5e-4",
    "--lambda_lpips", "10",
    "--lambda_l2", "10",
    "--lambda_gan", str(a.lambda_gan),
    "--lambda_clipsim", str(a.lambda_clip),
    "--input_dropout", "0",
    "--mask_dropout", "0",
    "--seed", str(a.seed),
    "--dataloader_num_workers", str(a.workers),
    "--report_to", "none",
])
train_ali2ali.main(args)
