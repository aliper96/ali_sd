"""
build_mvtec_all_splits.py — K splits of ALL of MVTec-AD (15 categories, 73 defect types) for the
full-dataset text-vs-colour experiment. Same stratification as D:/DATASET/build_aliaug_splits.py
(per (category, type), at least one sample on each side, test fraction 0.30, seed 1000+k), same prompt
texts (the paper's, D:/DATASET/_aliaug_prompts.json), images resized to 512 (the training resolution).

Difference: the conditioning image (C) follows the paper's HYBRID rule (Section "Unpaired Dataset
Scenario"):
  * textures (carpet, grid, leather, tile, wood): a random defect-free image of the category — any
    one is spatially compatible with a mask from another instance;
  * pose-varying objects: the defective image ITSELF with the masked region removed by inpainting
    (cv2.inpaint, Telea, on the mask dilated by 7 px), so the pose matches the mask but the defect is
    not available to copy. A random good image would put the mask on background.
Defect-free images come only from MVTec train/good, disjoint from test/ (no leakage).

    python prep/t23/build_mvtec_all_splits.py --mvtec <MVTec-AD root> --out <dir> --k 3
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import TEXTURES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mvtec", required=True)
ap.add_argument("--prompts", default=r"D:\DATASET\_aliaug_prompts.json")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--test-frac", type=float, default=0.30)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--seed0", type=int, default=1000)
a = ap.parse_args()
MV = Path(a.mvtec)
texts = json.load(open(a.prompts, encoding="utf-8"))


def load_rgb(p):
    return Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)


def load_mask(p):
    return Image.open(p).convert("L").resize((a.size, a.size), Image.NEAREST)


def inpaint_masked(img, mask):
    m = (np.asarray(mask) > 127).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, m, 7, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


inv = defaultdict(list)
for cat in sorted(texts):
    gd = MV / cat / "ground_truth"
    for ddir in sorted(x for x in gd.iterdir() if x.is_dir()):
        if ddir.name not in texts[cat]:
            print(f"  warning: no text for {cat}/{ddir.name}, skipped")
            continue
        for m in sorted(ddir.glob("*.png")):
            b = MV / cat / "test" / ddir.name / m.name.replace("_mask", "")
            if b.exists():
                inv[(cat, ddir.name)].append((m, b))
print(sum(len(v) for v in inv.values()), "defect samples,", len(inv), "(category, type) pairs")

for k in range(a.k):
    rng = random.Random(a.seed0 + k)
    O = Path(a.out) / f"split_{k}"
    pr = {"train": {}, "test": {}}
    for (cat, typ), pairs in sorted(inv.items()):
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * a.test_frac))
        n_test = min(n_test, len(idx) - 1) if len(idx) > 1 else 0
        test_i = set(idx[:n_test])
        goods = sorted((MV / cat / "train" / "good").glob("*.png"))
        for j, (m, b) in enumerate(pairs):
            fold = "test" if j in test_i else "train"
            name = f"{cat}_{typ}_{j:03d}.png"
            for sub in "ABC":
                (O / f"{fold}_{sub}").mkdir(parents=True, exist_ok=True)
            mask, img = load_mask(m), load_rgb(b)
            mask.save(O / f"{fold}_A" / name)
            img.save(O / f"{fold}_B" / name)
            clean = load_rgb(rng.choice(goods)) if cat in TEXTURES else inpaint_masked(img, mask)
            clean.save(O / f"{fold}_C" / name)
            pr[fold][name] = texts[cat][typ]
    for fold in ("train", "test"):
        json.dump(pr[fold], open(O / f"{fold}_prompts.json", "w", encoding="utf-8"), indent=1)
    assert not set(pr["train"]) & set(pr["test"])
    print(f"split_{k}: train {len(pr['train'])}  test {len(pr['test'])}")
