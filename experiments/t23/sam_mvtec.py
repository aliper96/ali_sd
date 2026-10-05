"""
sam_mvtec.py — SAM 3 on held-out synthetic (or real) defect images of one MVTec-AD category (protocol: PLAN.md,
"SAM 3 study"). For each image and its requested mask, prompts SAM 3 with every defect-type name of the category and
"defect"; keeps instances with score >= --thr. Writes one record per image: IoU of the union with the requested mask,
coverage of the mask, bleed outside the 15-px-dilated mask, hit flag, and the SAM label (type whose best instance
overlapping the mask scores highest).

    python sam_mvtec.py --split D:/ad/aliaug_cat/<cat>/split_0 --images <dir> --out <json> [--real]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--images", required=True, help="dir with <name>.png (synthetic pool, or test_B for real)")
ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--sam-code", default="C:/Users/aliha/Desktop/PHD/sam3")
a = ap.parse_args()
sys.path.insert(0, a.sam_code)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvtec_names import parse  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

names = sorted(json.load(open(os.path.join(a.split, "test_prompts.json"), encoding="utf-8")))
names = [n for n in names if os.path.exists(os.path.join(a.images, n))]
cat = parse(names[0])[0]
types = sorted({parse(n)[1] for f in ("train_prompts.json", "test_prompts.json")
                for n in json.load(open(os.path.join(a.split, f), encoding="utf-8"))})
phrases = {t: t.replace("_", " ") for t in types}
# PLAN.md amendment 2026-10-04: SAM 3 ignores most MVTec type names and "defect"; generic vocabulary for WHERE only
VOCAB = ["hole", "crack", "scratch", "stain", "dent", "spot", "contamination", "broken", "missing part", "cut", "tear",
         "bent", "hair", "thread", "fold"]
proc = Sam3Processor(build_sam3_image_model())
K = np.ones((31, 31), np.uint8)
recs = []
for n in names:
    im = Image.open(os.path.join(a.images, n)).convert("RGB").resize((512, 512), Image.BICUBIC)
    m = np.asarray(Image.open(os.path.join(a.split, "test_A", n)).convert("L").resize((512, 512), Image.NEAREST)) > 25
    dil = cv2.dilate(m.astype(np.uint8), K) > 0
    st = proc.set_image(im)
    union = np.zeros(m.shape, bool)
    best = {}
    for key, ph in list(phrases.items()) + [("_defect", "defect")] + [("_v_" + w, w) for w in VOCAB]:
        out = proc.set_text_prompt(state=st, prompt=ph)
        if out["masks"] is None or len(out["scores"]) == 0:
            continue
        ms = out["masks"].squeeze(1).float().cpu().numpy() > 0.5
        sc = out["scores"].float().cpu().numpy()
        for mi, s in zip(ms, sc):
            if s < a.thr:
                continue
            if mi.shape != m.shape:
                mi = cv2.resize(mi.astype(np.uint8), m.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            union |= mi
            if not key.startswith("_") and (mi & m).sum() > 0:
                best[key] = max(best.get(key, 0.0), float(s))
    inter, uni = (union & m).sum(), (union | m).sum()
    recs.append({"name": n, "type": parse(n)[1], "iou": float(inter / uni) if uni else 0.0,
                 "cover": float(inter / max(m.sum(), 1)), "bleed": float((union & ~dil).sum() / max(union.sum(), 1)),
                 "pred_area": int(union.sum()), "hit": bool(inter / max(m.sum(), 1) >= 0.10),
                 "sam_label": max(best, key=best.get) if best else None, "best_scores": best})
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"category": cat, "images": a.images, "thr": a.thr, "types": types, "records": recs}, open(a.out, "w"), indent=1)
h = np.mean([r["hit"] for r in recs]); lab = np.mean([r["sam_label"] == r["type"] for r in recs])
print(f"{cat}: {len(recs)} images | hit {h:.2f} | IoU {np.mean([r['iou'] for r in recs]):.2f} | SAM label = mask type {lab:.2f}")
