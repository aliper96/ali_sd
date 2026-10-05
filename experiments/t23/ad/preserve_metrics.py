"""
preserve_metrics.py — does the generator keep the input image outside the mask, and how sharp is it?

For every held-out (test-fold) sample of split 0 and every generator (AD released ckpt, Ali-AUG text, Ali-AUG
colour), compares the generated image with the CLEAN INPUT it was given, outside the defect mask dilated by
15 px (so the defect and its immediate border are excluded):
  psnr_out  : PSNR outside the mask (higher = background better preserved)
  mae_out   : mean absolute error outside the mask, 0-255
  sharp     : variance of the Laplacian outside the mask, generated / clean (1 = as sharp as the input,
              < 1 = blurrier)
Computed at 512 px (the size used downstream) and at 256 px (AD's native resolution; everything downsampled),
so AD's 256 -> 512 upsampling is not the only explanation of its numbers.

    python preserve_metrics.py --cat-root D:/ad/aliaug_cat --ad D:/ad/judge_syn --ours D:/ad/ours_heldout \
        --out C:/Users/aliha/Desktop/PHD/t23_runs/live/preserve_metrics.json
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cat-root", required=True)
ap.add_argument("--ad", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

KER = np.ones((31, 31), np.uint8)  # 15-px dilation radius


def load(p, size):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) if im.shape[0] != size else im


def metrics(gen, clean, out_mask):
    g = gen.astype(np.float32)
    c = clean.astype(np.float32)
    d = (g - c)[out_mask]
    mse = float((d ** 2).mean())
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[out_mask].var()
    return {"psnr_out": 10 * np.log10(255 ** 2 / max(mse, 1e-6)), "mae_out": float(np.abs(d).mean()),
            "sharp": float(lap(gen) / max(lap(clean), 1e-6))}


res = defaultdict(lambda: defaultdict(list))
for cat in sorted(os.listdir(a.cat_root)):
    sp = os.path.join(a.cat_root, cat, "split_0")
    gens = {"AnomalyDiffusion": os.path.join(a.ad, f"ad_heldout_{cat}"),
            "Ali-AUG text": os.path.join(a.ours, f"cattext_heldout_{cat}"),
            "Ali-AUG colour": os.path.join(a.ours, f"catcolor_heldout_{cat}")}
    if not all(os.path.isdir(g) for g in gens.values()):
        continue
    names = sorted(n for n in os.listdir(os.path.join(sp, "test_A")) if all(os.path.exists(os.path.join(g, n)) for g in gens.values()))
    for size in (512, 256):
        for n in names:
            m = load(os.path.join(sp, "test_A", n), size).max(axis=2) > 25
            out_mask = cv2.dilate(m.astype(np.uint8), KER if size == 512 else KER[:16, :16]) == 0
            if out_mask.sum() < 100:
                continue
            clean = load(os.path.join(sp, "test_C", n), size)
            for meth, gdir in gens.items():
                for k, v in metrics(load(os.path.join(gdir, n), size), clean, out_mask).items():
                    res[f"{size}"][(cat, meth, k)].append(v)

summary = {}
for size, d in res.items():
    per_cat = defaultdict(dict)
    for (cat, meth, k), vals in d.items():
        per_cat[cat].setdefault(meth, {})[k] = float(np.mean(vals))
    overall = defaultdict(dict)
    for meth in ("AnomalyDiffusion", "Ali-AUG text", "Ali-AUG colour"):
        for k in ("psnr_out", "mae_out", "sharp"):
            overall[meth][k] = float(np.mean([per_cat[c][meth][k] for c in per_cat]))
    summary[size] = {"per_category": per_cat, "mean_over_categories": overall, "n_categories": len(per_cat)}
json.dump(summary, open(a.out, "w"), indent=1)
for size in summary:
    print(f"--- {size}px, {summary[size]['n_categories']} categories (mean of per-category means)")
    for meth, v in summary[size]["mean_over_categories"].items():
        print(f"{meth:18s} PSNR_out {v['psnr_out']:.2f} dB | MAE_out {v['mae_out']:.2f} | sharpness ratio {v['sharp']:.3f}")
