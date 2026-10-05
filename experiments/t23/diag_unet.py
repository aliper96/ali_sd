"""
diag_unet.py — does the U-Net (the only path of the text prompt) affect the released model's output?

Same inputs, same seed, four generations with a trained checkpoint:
  own prompt | another prompt | U-Net output zeroed | U-Net output x10
and the scheduler state. If 'U-Net zeroed' equals 'own prompt', the U-Net is inert and the text
cannot matter. Also measures the coefficient that scheduler.step() applies to the U-Net output.
"""
import argparse
import sys
import types
import importlib.machinery

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", default="tile_crack_000.png")
ap.add_argument("--fix-sched", action="store_true")
a = ap.parse_args()
try:
    import wandb  # noqa: F401
except ImportError:
    w = types.ModuleType("wandb"); w.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = w
import os
import torch
import torchvision.transforms.functional as F
from PIL import Image

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
    cur = mod.state_dict(); cur.update(sd[key]); mod.load_state_dict(cur)
m.set_eval()
s = m.sched
print("scheduler num_inference_steps:", s.num_inference_steps, "| timesteps:", getattr(s, "timesteps", None)[:3])
t = 999
ab = s.alphas_cumprod[t].item()
prev = s.previous_timestep(torch.tensor(t)).item() if hasattr(s, "previous_timestep") else None
print("step() goes from t=999 to t=", prev)

T = build_transform("resized_crop_512")
mask = F.to_tensor(T(Image.open(os.path.join(a.split, "test_A", a.name)).convert("RGB"))).unsqueeze(0).cuda()
clean = F.normalize(F.to_tensor(T(Image.open(os.path.join(a.split, "test_C", a.name)).convert("RGB"))),
                    [0.5], [0.5]).unsqueeze(0).cuda()

orig_forward = m.unet.forward


def gen(prompt, scale=1.0):
    def fwd(*x, **k):
        out = orig_forward(*x, **k)
        out.sample = out.sample * scale
        return out
    m.unet.forward = fwd
    torch.manual_seed(0)
    with torch.no_grad():
        y = m(mask, clean, prompt=prompt)
    m.unet.forward = orig_forward
    return y


base = gen("Add a crack.")
for label, y in [("another prompt ('Add oil.')", gen("Add oil.")),
                 ("U-Net output x0", gen("Add a crack.", 0.0)),
                 ("U-Net output x10", gen("Add a crack.", 10.0))]:
    d = (y - base).abs() * 127.5
    print(f"{label:28s} max |diff| {d.max().item():7.3f}  mean {d.mean().item():7.4f}  (grey levels)")
