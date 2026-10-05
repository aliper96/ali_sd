"""
gen_syn.py — generate the synthetic pool for ONE split of the Tables 2-3 re-run.

For every TRAINING triplet of the split (mask train_A, clean image train_C, prompt)
the trained generator produces one image, so N_syn = N_real_train (NAS 1:1). Only the
training fold is touched (leakage rules L1-L3 of RESULTS_SPEC.md). Uses the RELEASED
model code (ali_sd_public/ali2ali.py) with the same preprocessing as its trainer
(resized_crop_512). Each image is generated with a fixed per-sample seed (S5).

The released Ali2Ali only loads checkpoints from a hard-coded path, so the weights are
loaded here the same way it does it (LoRA/skip/conv_in keys copied into the state dict).

Output: <out>/<name>.png (512x512) and <out>/prompts.json (name -> prompt used).
"""
import argparse
import json
import os
import sys
import types
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-mask", type=int, default=1,
                help="E3 ratio sweep: images per training mask. j=0 uses the mask's own clean "
                     "image; j>0 use other clean images of the SAME training fold (fixed seed). "
                     "Extra files are named <stem>__c<j>.png")
ap.add_argument("--fold", choices=["train", "test"], default="train",
                help="E2 only: 'test' generates from HELD-OUT masks the generator never saw, to "
                     "measure label fidelity without memorisation. Never used for training data.")
ap.add_argument("--swap", action="store_true",
                help="E2 only: request the NEXT defect class (cyclic) instead of the mask's own: "
                     "text variant -> that class's prompt; colour variant -> mask repainted in that "
                     "class's colour. Tests whether the label channel or the mask shape decides.")
ap.add_argument("--fix-sched", action="store_true", help="see fixsched.py; must match training")
ap.add_argument("--no-skip", action="store_true", help="see ablations.py; must match training")
a = ap.parse_args()

try:  # the model code does not use wandb, but diffusers probes find_spec("wandb")
    import wandb  # noqa: F401
except ImportError:
    import importlib.machinery
    _wb = types.ModuleType("wandb")
    _wb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = _wb
import torch  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, a.code)
if a.fix_sched:  # must match how the checkpoint was trained
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixsched  # noqa: E402
    fixsched.apply()
from ali2ali import Ali2Ali  # noqa: E402
if a.no_skip:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ablations  # noqa: E402
    ablations.apply_no_skip()
from utils import build_transform  # noqa: E402

m = Ali2Ali(lora_rank_unet=8, lora_rank_vae=4)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
assert (sd["rank_unet"], sd["rank_vae"]) == (8, 4), (sd["rank_unet"], sd["rank_vae"])
for mod, key in ((m.unet, "state_dict_unet"), (m.vae, "state_dict_vae")):
    cur = mod.state_dict()
    missing = [k for k in sd[key] if k not in cur]
    assert not missing, missing[:3]
    cur.update(sd[key])
    mod.load_state_dict(cur)
m.set_eval()

import random  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvtec_names import TEXTURES, colour, next_type, parse, types_by_category  # noqa: E402

T = build_transform("resized_crop_512")
f = a.fold
prompts = json.load(open(os.path.join(a.split, f"{f}_prompts.json"), encoding="utf-8"))
colour_variant = len(set(prompts.values())) == 1          # constant prompt -> class is the colour
names = sorted(os.listdir(os.path.join(a.split, f"{f}_A")))
# types per category from BOTH folds, so a swap can request a type absent from this fold
all_names = names + [n for n in os.listdir(os.path.join(a.split, "train_A"))]
types_of = types_by_category(all_names)
type_prompt = {}  # text variant: (category, type) -> its prompt (from both folds)
for fold in ("train", "test"):
    fp = os.path.join(a.split, f"{fold}_prompts.json")
    if os.path.exists(fp):
        for n, pmt in json.load(open(fp, encoding="utf-8")).items():
            type_prompt[parse(n)] = pmt
os.makedirs(a.out, exist_ok=True)
used, labels = {}, {}
for name in names:
    cat, t_mask = parse(name)
    t_req = next_type(cat, t_mask, types_of[cat]) if a.swap else t_mask
    mimg = Image.open(os.path.join(a.split, f"{f}_A", name)).convert("RGB")
    if colour_variant and t_req != t_mask:  # same geometry, repainted in the requested colour
        geo = np.asarray(mimg).max(axis=2) > 25
        arr = np.zeros((*geo.shape, 3), np.uint8)
        arr[geo] = colour(cat, t_req, types_of[cat])
        mimg = Image.fromarray(arr)
    mask = F.to_tensor(T(mimg))
    prompt = prompts[name] if colour_variant else type_prompt[(cat, t_req)]
    # extra clean images (E3): same category, and only for textures — for pose-varying objects the
    # clean input is this sample's own inpainted image, so other images would not align with the mask
    others = [n for n in names if n != name and parse(n)[0] == cat] if cat in TEXTURES else []
    random.Random(zlib.crc32(name.encode())).shuffle(others)
    for j in range(a.per_mask):
        src = others[j - 1] if 0 < j <= len(others) else name
        clean = F.to_tensor(T(Image.open(os.path.join(a.split, f"{f}_C", src)).convert("RGB")))
        clean = F.normalize(clean, mean=[0.5], std=[0.5])
        oname = name if j == 0 else f"{name[:-4]}__c{j}.png"
        torch.manual_seed(zlib.crc32(oname.encode()))
        with torch.no_grad():
            out = m(mask.unsqueeze(0).cuda(), clean.unsqueeze(0).cuda(), prompt=prompt)
        F.to_pil_image((out[0] * 0.5 + 0.5).clamp(0, 1).cpu()).save(os.path.join(a.out, oname))
        used[oname] = prompt
        labels[oname] = {"requested": t_req, "mask": t_mask, "category": cat}
json.dump(labels, open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8"), indent=1)
json.dump(used, open(os.path.join(a.out, "prompts.json"), "w", encoding="utf-8"), indent=1)
print(f"generated {len(used)} images -> {a.out}")
