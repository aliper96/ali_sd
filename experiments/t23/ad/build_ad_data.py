"""
build_ad_data.py — MVTec-style folder for training the AnomalyDiffusion baseline on OUR train fold only.

AnomalyDiffusion's loader reads <root>/<cat>/test/<type>/NNN.png + <root>/<cat>/ground_truth/<type>/NNN_mask.png
and trains on the lowest-ID third (patched in patch_ad.py to take every file when AD_ALL=1). Here those
folders contain ONLY the samples of the train fold of split 0 (aliaug_cat/<cat>/split_0: train_B = real
defective image, train_A = mask), so the generator never sees a test-fold image — the same rule as Ali-AUG.
Also writes name-anomaly.txt (cat+type list) for the categories of the per-category study.

    python3 build_ad_data.py --cat-root /mnt/scratch/t23/aliaug_cat --out /mnt/scratch/ad/mvtec_ours \
        --name-list /mnt/scratch/ad/anomalydiffusion/name-anomaly.txt
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvtec_names import parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cat-root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name-list", required=True)
ap.add_argument("--split", default="split_0")
a = ap.parse_args()

pairs = []
manifest = {}
for cat in sorted(os.listdir(a.cat_root)):
    sp = Path(a.cat_root) / cat / a.split
    if not (sp / "train_prompts.json").exists():
        continue
    names = sorted(json.load(open(sp / "train_prompts.json", encoding="utf-8")))
    by_type = defaultdict(list)
    for n in names:
        by_type[parse(n)[1]].append(n)
    for typ, ns in sorted(by_type.items()):
        di = Path(a.out) / cat / "test" / typ
        dm = Path(a.out) / cat / "ground_truth" / typ
        di.mkdir(parents=True, exist_ok=True)
        dm.mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(ns):
            Image.open(sp / "train_B" / n).convert("RGB").save(di / f"{i:03d}.png")
            m = np.asarray(Image.open(sp / "train_A" / n).convert("RGB")).max(axis=2) > 25
            Image.fromarray((m * 255).astype(np.uint8)).save(dm / f"{i:03d}_mask.png")
            manifest[f"{cat}/{typ}/{i:03d}"] = n
        pairs.append(f"{cat}+{typ}")
    print(f"{cat:12s} {len(names):3d} train samples, {len(by_type)} types")
open(a.name_list, "w").write("\n".join(pairs))
json.dump(manifest, open(Path(a.out) / "manifest.json", "w"), indent=1)
print(len(pairs), "cat+type pairs ->", a.name_list)
