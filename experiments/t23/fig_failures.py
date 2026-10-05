"""
fig_failures.py — failure-case figure for review theme Q, from the E2 judge (no GPU).

Two panels, text variant, held-out (test-fold) masks never seen by the generator:
  A. own class requested, independent judge assigns ANOTHER class  -> wrong weak label
  B. swap test: prompt requests the next class; the judge sees the class of the MASK SHAPE
     -> the prompt was ignored
Only cases on which all judge seeds agree are shown (not artefacts of one classifier run).
Columns: clean input | mask | generated | real defect for this mask (its true class).

    python prep/t23/fig_failures.py --judge C:/Users/aliha/Desktop/PHD/t23_runs/judge \
        --splits D:/DATASET/aliaug_splits --out revision_assets/fig_failures_text.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--judge", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="text")
ap.add_argument("--rows", type=int, default=3)
a = ap.parse_args()


def robust(mode):
    """(split, name) -> (requested, judged) where every judge seed disagrees the same way."""
    votes = defaultdict(list)
    files = glob.glob(os.path.join(a.judge, "results_judge", f"{a.variant}_{mode}_split*_seed*.json"))
    seeds_per_split = defaultdict(int)
    for f in files:
        d = json.load(open(f))
        k = int(os.path.basename(f).split("_split")[1].split("_")[0])
        seeds_per_split[k] += 1
        for x in d["disagreements"]:
            votes[(k, x["name"])].append((x["requested"], x["judged"]))
    out = {}
    for (k, n), v in votes.items():
        if len(v) == seeds_per_split[k] and len(set(v)) == 1:
            out[(k, n)] = v[0]
    return out


def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


C, PAD, LAB, TIT = 200, 6, 36, 34
panels = [("A. Held-out mask, own class requested: judge sees another class (wrong weak label)",
           "heldout"),
          ("B. Swap test: prompt asks for another class; output follows the mask shape",
           "swap")]
rows = []
for title, mode in panels:
    cases, seen = [], set()
    for (k, n), (req, jud) in sorted(robust(mode).items()):  # one row per mask class
        cls = n[5:].rsplit("_", 1)[0]
        if cls not in seen:
            seen.add(cls)
            cases.append(((k, n), (req, jud)))
    cases = cases[: a.rows]
    rows.append(("title", title))
    for (k, n), (req, jud) in cases:
        sp = os.path.join(a.splits, f"split_{k}")
        ims = [Image.open(os.path.join(sp, "test_C", n)), Image.open(os.path.join(sp, "test_A", n)),
               Image.open(os.path.join(a.judge, "judge_syn", f"{a.variant}_{mode}_split_{k}", n)),
               Image.open(os.path.join(sp, "test_B", n))]
        rows.append(("row", (ims, f"requested: {req}   |   judge: {jud}   ({n[5:-4]}, split {k})")))

W = 4 * C + 5 * PAD
H = sum(TIT if r[0] == "title" else C + LAB + PAD for r in rows) + PAD + 28
fig = Image.new("RGB", (W, H), (250, 250, 250))
d = ImageDraw.Draw(fig)
y = PAD
for j, h in enumerate(["clean input", "mask", "generated", "real defect (this mask)"]):
    d.text((PAD + j * (C + PAD) + 4, y), h, fill=(20, 20, 20), font=font(15))
y += 28
for kind, content in rows:
    if kind == "title":
        d.text((PAD, y + 8), content, fill=(150, 20, 20), font=font(16))
        y += TIT
        continue
    ims, lab = content
    for j, im in enumerate(ims):
        fig.paste(im.convert("RGB").resize((C, C)), (PAD + j * (C + PAD), y))
    d.text((PAD + 4, y + C + 8), lab, fill=(20, 20, 20), font=font(15))
    y += C + LAB + PAD
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.save(a.out)
print("saved", a.out, {m: len(robust(m)) for _, m in panels})
