"""
build_cat_splits.py — per-category splits for the per-category LoRA study, cut from the full-MVTec split.

For every category (except those listed in --skip), writes --dst/<category>/split_<k> containing only that
category's samples of --src/split_<k>: {train,test}_{A,B,C} as per-file symlinks and the filtered
{train,test}_prompts.json. Same samples, same hybrid clean inputs and same train/test assignment as the
full-MVTec split, so per-category and single-adapter results are directly comparable.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import CATEGORIES, parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="full-MVTec splits dir (aliaug_splits_all)")
ap.add_argument("--dst", required=True)
ap.add_argument("--k", type=int, nargs="+", default=[0])
ap.add_argument("--skip", nargs="*", default=["tile"])
a = ap.parse_args()

for k in a.k:
    sp = Path(a.src) / f"split_{k}"
    for cat in CATEGORIES:
        if cat in a.skip:
            continue
        out = Path(a.dst) / cat / f"split_{k}"
        n = {}
        for fold in ("train", "test"):
            pr = json.load(open(sp / f"{fold}_prompts.json", encoding="utf-8"))
            keep = {name: p for name, p in pr.items() if parse(name)[0] == cat}
            for sub in "ABC":
                d = out / f"{fold}_{sub}"
                d.mkdir(parents=True, exist_ok=True)
                for name in keep:
                    link = d / name
                    if not link.exists():
                        os.symlink((sp / f"{fold}_{sub}" / name).resolve(), link)
            json.dump(keep, open(out / f"{fold}_prompts.json", "w", encoding="utf-8"), indent=1)
            n[fold] = len(keep)
        print(f"{cat:12s} split_{k}: train {n['train']:3d}  test {n['test']:3d}")
