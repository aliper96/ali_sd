"""
time_infer.py — inference cost of the Ali-AUG generator (one step) for the efficiency table.

Loads a trained per-category checkpoint exactly as gen_syn.py does, then times N generations at 512 px,
batch 1, after W warm-up runs (CUDA synchronised around each call). Reports mean / std / median seconds per
image, images per second, peak GPU memory (max_memory_allocated) and the parameter counts (total, trainable
in the checkpoint). Must run on an otherwise IDLE GPU; the script refuses to time if another process is using
more than --max-busy-mb of GPU memory.

    python3 time_infer.py --code ali_sd_public --fix-sched --split <split_0> --ckpt model_10000.pkl \
        --out results_timing/aliaug_<carrier>_<cat>.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import types

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fix-sched", action="store_true")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--max-busy-mb", type=int, default=2000)
a = ap.parse_args()

used = [int(x) for x in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()]
if used[0] > a.max_busy_mb:
    sys.exit(f"GPU not idle ({used[0]} MB in use): timing would be contaminated")

try:
    import wandb  # noqa: F401
except ImportError:
    import importlib.machinery
    _wb = types.ModuleType("wandb")
    _wb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = _wb
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, a.code)
if a.fix_sched:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixsched  # noqa: E402
    fixsched.apply()
from ali2ali import Ali2Ali  # noqa: E402
from utils import build_transform  # noqa: E402

m = Ali2Ali(lora_rank_unet=8, lora_rank_vae=4)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
for mod, key in ((m.unet, "state_dict_unet"), (m.vae, "state_dict_vae")):
    cur = mod.state_dict()
    cur.update(sd[key])
    mod.load_state_dict(cur)
m.set_eval()
n_total = sum(p.numel() for p in m.parameters())
n_ckpt = sum(v.numel() for k in ("state_dict_unet", "state_dict_vae") for v in sd[k].values())

T = build_transform("resized_crop_512")
prompts = json.load(open(os.path.join(a.split, "train_prompts.json"), encoding="utf-8"))
name = sorted(prompts)[0]
mask = F.to_tensor(T(Image.open(os.path.join(a.split, "train_A", name)).convert("RGB"))).unsqueeze(0).cuda()
clean = F.normalize(F.to_tensor(T(Image.open(os.path.join(a.split, "train_C", name)).convert("RGB"))),
                    mean=[0.5], std=[0.5]).unsqueeze(0).cuda()

torch.cuda.reset_peak_memory_stats()
times = []
with torch.no_grad():
    for i in range(a.warmup + a.n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(mask, clean, prompt=prompts[name])
        torch.cuda.synchronize()
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
t = np.array(times)
res = {"method": "Ali-AUG (one step)", "ckpt": a.ckpt, "resolution": 512, "batch": 1, "steps": 1,
       "n": a.n, "warmup": a.warmup, "s_per_img_mean": float(t.mean()), "s_per_img_std": float(t.std()),
       "s_per_img_median": float(np.median(t)), "img_per_s": float(1 / t.mean()),
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
       "params_total_M": n_total / 1e6, "params_in_checkpoint_M": n_ckpt / 1e6,
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
