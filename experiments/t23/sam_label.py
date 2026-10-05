"""
sam_label.py — E5: the automatic step of the labelling workflow (generate -> SAM -> human check).

For every image of a folder (synthetic defects, or the REAL test images as reference) and the mask it
should contain, SAM 3 is prompted with text:
  * WHERE — prompt "defect": union of predicted instances vs the requested mask -> IoU, and the fraction
    of predicted defect area that falls OUTSIDE the requested mask (bleed / misplaced defect);
  * WHAT — one prompt per class name; the auto-label is the class whose instances score highest inside
    the requested mask. Compared with the class the image is supposed to carry.
Run it on the real test images first: SAM 3 was not trained on tile defects, so the real-image numbers
are the reference (ceiling) for everything measured on synthetic images.

    python prep/t23/sam_label.py --images <dir> --masks <dir with same file names> \
        [--labels labels.json] --out result.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
# natural-language names for SAM's text encoder
PHRASE = {"crack": "crack", "glue_strip": "glue strip", "gray_stroke": "gray stroke",
          "oil": "oil stain", "rough": "rough patch"}

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--masks", required=True)
ap.add_argument("--labels", default=None, help="labels.json from gen_syn.py (requested class)")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--thr", type=float, default=0.3, help="score threshold for kept instances")
a = ap.parse_args()

# Needs triton (Linux): SAM 3 imports it at module load. Run on the A100, not on Windows.
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

proc = Sam3Processor(build_sam3_image_model())
lab = json.load(open(a.labels)) if a.labels else {}
files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() == ".png")
if a.limit:
    files = files[: a.limit]


def dtype(n):
    return re.match(r"tile_(.+)_\d+\.png$", n).group(1)


def run(state, phrase, shape):
    out = proc.set_text_prompt(state=state, prompt=phrase)
    masks, scores = out["masks"], out["scores"]
    if masks is None or len(scores) == 0:
        return np.zeros(shape, bool), []
    m = masks.squeeze(1).float().cpu().numpy() > 0.5 if torch.is_tensor(masks) else np.asarray(masks) > 0.5
    s = scores.float().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    keep = s >= a.thr
    union = m[keep].any(0) if keep.any() else np.zeros(shape, bool)
    return union, list(zip(m, s))


rows = []
for f in files:
    img = Image.open(f).convert("RGB")
    req = np.asarray(Image.open(Path(a.masks) / f.name).convert("L").resize(img.size)) > 127
    target = lab.get(f.name, {}).get("requested", dtype(f.name))
    st = proc.set_image(img)
    where, _ = run(st, "defect", req.shape)
    inter, uni = (where & req).sum(), (where | req).sum()
    per_class = {}
    for t in TYPES:
        _, inst = run(st, PHRASE[t], req.shape)
        # best score among instances that overlap the requested mask by >= 30 % of their area
        best = 0.0
        for m, s in inst:
            if m.sum() and (m & req).sum() / m.sum() >= 0.3:
                best = max(best, float(s))
        per_class[t] = best
    auto = max(per_class, key=per_class.get) if max(per_class.values()) > 0 else None
    rows.append(dict(name=f.name, target=target, iou=float(inter / uni) if uni else 0.0,
                     outside=float((where & ~req).sum() / where.sum()) if where.sum() else None,
                     found=bool(where.sum()), auto_label=auto, scores=per_class))
    print(f"{f.name:26s} target={target:11s} auto={str(auto):11s} IoU={rows[-1]['iou']:.2f}", flush=True)

found = [r for r in rows if r["found"]]
summary = dict(
    n=len(rows), detected=len(found) / len(rows),
    mean_iou=float(np.mean([r["iou"] for r in rows])),
    mean_outside=float(np.mean([r["outside"] for r in found])) if found else None,
    auto_label_accuracy=float(np.mean([r["auto_label"] == r["target"] for r in rows])),
    no_auto_label=float(np.mean([r["auto_label"] is None for r in rows])),
)
json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
