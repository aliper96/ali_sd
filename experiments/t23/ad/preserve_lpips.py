"""
preserve_lpips.py — perceptual background preservation: LPIPS (AlexNet, spatial map) between the generated image
and the clean INPUT, averaged only outside the defect mask dilated by 15 px. Unlike pixel PSNR it tolerates
small shifts of a repeated texture, so it measures whether the background still LOOKS the same.
All images at 256 px (AD's native size; Ali-AUG downsampled), so resolution is not what is being measured.
Lower = background better preserved. Reference row: two DIFFERENT real defect-free images of the same texture
category would be "different texture instance" (not computed here: only generated vs its own input).

    python preserve_lpips.py --cat-root D:/ad/aliaug_cat --ad D:/ad/judge_syn --ours D:/ad/ours_heldout \
        --out C:/Users/aliha/Desktop/PHD/t23_runs/live/preserve_lpips.json
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import lpips
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cat-root", required=True)
ap.add_argument("--ad", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = 256
net = lpips.LPIPS(net="alex", spatial=True).cuda().eval()


def tens(p):
    im = cv2.cvtColor(cv2.resize(cv2.imread(p), (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(im).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1


res = defaultdict(dict)
for cat in sorted(os.listdir(a.cat_root)):
    sp = os.path.join(a.cat_root, cat, "split_0")
    gens = {"AnomalyDiffusion": os.path.join(a.ad, f"ad_heldout_{cat}"),
            "Ali-AUG text": os.path.join(a.ours, f"cattext_heldout_{cat}"),
            "Ali-AUG colour": os.path.join(a.ours, f"catcolor_heldout_{cat}")}
    if not all(os.path.isdir(g) for g in gens.values()):
        continue
    names = sorted(n for n in os.listdir(os.path.join(sp, "test_A"))
                   if all(os.path.exists(os.path.join(g, n)) for g in gens.values()))
    vals = defaultdict(list)
    for n in names:
        m = cv2.resize(cv2.imread(os.path.join(sp, "test_A", n)), (S, S), interpolation=cv2.INTER_NEAREST).max(axis=2) > 25
        out = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0  # ~15 px at 512 = 7.5 at 256
        if out.sum() < 100:
            continue
        outm = torch.from_numpy(out).cuda()
        clean = tens(os.path.join(sp, "test_C", n))
        with torch.no_grad():
            for meth, g in gens.items():
                d = net(tens(os.path.join(g, n)), clean)[0, 0]
                vals[meth].append(float(d[outm].mean()))
    res[cat] = {meth: float(np.mean(v)) for meth, v in vals.items()}
    print(f"{cat:11s} " + " | ".join(f"{m} {v:.3f}" for m, v in res[cat].items()))
mean = {m: float(np.mean([res[c][m] for c in res])) for m in next(iter(res.values()))}
objs = [c for c in res if c not in ("carpet", "grid", "leather", "wood", "tile")]
tex = [c for c in res if c in ("carpet", "grid", "leather", "wood", "tile")]
grp = {g: {m: float(np.mean([res[c][m] for c in cs])) for m in mean} for g, cs in (("objects", objs), ("textures", tex))}
json.dump({"per_category": res, "mean": mean, "groups": grp, "size": S}, open(a.out, "w"), indent=1)
print("MEAN", {m: round(v, 3) for m, v in mean.items()})
for g, v in grp.items():
    print(g.upper(), {m: round(x, 3) for m, x in v.items()})
