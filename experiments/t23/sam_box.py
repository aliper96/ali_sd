"""
sam_box.py — the SAM step of generate -> SAM -> human check (PLAN.md, SAM 3 study, amendment 2).

For every image of a pool (synthetic images of one category, or the real test defects) and its requested mask,
SAM 3 is prompted with the bounding box of the requested mask (padded 10 %, at least 16 px, positive box). The SAM
mask is the returned instance with the highest IoU with the requested mask. Writes one record per image (IoU with
the requested mask, SAM score, area ratio, accepted = IoU >= --accept) and, for ACCEPTED images, the SAM mask as a
0/255 PNG with the image's name in --mask-out (consumed by build_yolo.py --syn-masks --drop-missing).
For real images (--gt) the requested mask is the ground truth, so the IoU measures how well SAM recovers a real
defect from a box.

    python sam_box.py --split D:/ad/aliaug_cat/<cat>/split_0 --fold train|test --images <dir> --out <json>
                      [--mask-out <dir>] [--accept 0.25]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--fold", choices=["train", "test"], required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mask-out", default=None)
ap.add_argument("--accept", type=float, default=0.25)
ap.add_argument("--sam-code", default="C:/Users/aliha/Desktop/PHD/sam3")
a = ap.parse_args()
sys.path.insert(0, a.sam_code)
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

names = sorted(json.load(open(os.path.join(a.split, f"{a.fold}_prompts.json"), encoding="utf-8")))
pool = set(os.listdir(a.images))
# extra pools (ratio sweep) are named <stem>__c<j>.png; this study uses the 1-per-mask pools only
names = [n for n in names if n in pool]
proc = Sam3Processor(build_sam3_image_model())
if a.mask_out:
    os.makedirs(a.mask_out, exist_ok=True)


def box_of(m, pad=0.10):
    ys, xs = np.where(m)
    H, W = m.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = max(x1 - x0, 16) * (1 + pad), max(y1 - y0, 16) * (1 + pad)
    return [((x0 + x1) / 2) / W, ((y0 + y1) / 2) / H, min(w / W, 1.0), min(h / H, 1.0)]


recs = []
for n in names:
    im = Image.open(os.path.join(a.images, n)).convert("RGB").resize((512, 512), Image.BICUBIC)
    m = np.asarray(Image.open(os.path.join(a.split, f"{a.fold}_A", n)).convert("L").resize((512, 512), Image.NEAREST)) > 25
    rec = {"name": n, "iou": 0.0, "score": 0.0, "area_ratio": 0.0, "n_inst": 0, "accepted": False}
    if m.sum() > 0:
        st = proc.set_image(im)
        out = proc.add_geometric_prompt(box=box_of(m), label=True, state=st)
        if out["masks"] is not None and len(out["scores"]) > 0:
            ms = out["masks"].squeeze(1).float().cpu().numpy() > 0.5
            sc = out["scores"].float().cpu().numpy()
            ious = [float((x & m).sum() / max((x | m).sum(), 1)) for x in ms]
            k = int(np.argmax(ious))
            rec.update(iou=ious[k], score=float(sc[k]), area_ratio=float(ms[k].sum() / m.sum()), n_inst=int(len(sc)))
            rec["accepted"] = bool(ious[k] >= a.accept)
            if rec["accepted"] and a.mask_out:
                Image.fromarray((ms[k] * 255).astype(np.uint8)).save(os.path.join(a.mask_out, n))
    recs.append(rec)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"split": a.split, "fold": a.fold, "images": a.images, "accept": a.accept, "records": recs}, open(a.out, "w"), indent=1)
print(f"{os.path.basename(a.images)}: {len(recs)} | mean IoU {np.mean([r['iou'] for r in recs]):.3f} | "
      f"accepted {np.mean([r['accepted'] for r in recs]):.2f}")
