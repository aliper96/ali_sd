"""make_pools.py — two control pools per category for the review of 2026-10-05.

  comp_<carrier>_<cat> : the Ali-AUG training pool with the conditioning input re-inserted outside the
                         requested mask (dilated by --dilate px): post-hoc compositing. Same labels.
  paste_<cat>          : non-generative copy-paste baseline. For every training mask the real defect
                         (train_B inside the mask) is Poisson-blended (cv2.seamlessClone) onto the
                         conditioning input train_C at the same position. For objects train_C is the
                         inpainted image of the same sample, so the result is close to the real image;
                         for textures it is another defect-free image. Labels = type of the mask.
Pools keep gen_syn's prompts.json/labels.json format so build_yolo.py and judge.py read them unchanged.
  python prep/t23/make_pools.py --cat-root D:/ad/aliaug_cat --pools D:/sam3/trainpools --out D:/pools
"""
import argparse, json, os, sys
import cv2, numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cat-root", required=True)
ap.add_argument("--pools", required=True, help="dir with <carrier>_syn_<cat>/ (gen_syn output)")
ap.add_argument("--out", required=True)
ap.add_argument("--dilate", type=int, default=15)
ap.add_argument("--cats", nargs="*", default=None)
ap.add_argument("--heldout", action="store_true",
                help="composite the held-out pools (<carrier>_heldout_<cat>, test fold) into --out/<carrier>_heldout_<cat> "
                     "for preserve_lpips.py / preserve_metrics.py; no copy-paste pool")
a = ap.parse_args()
cats = a.cats or sorted(d for d in os.listdir(a.cat_root) if os.path.isdir(os.path.join(a.cat_root, d)))
K = np.ones((a.dilate, a.dilate), np.uint8)


def load_rgb(p, size=512):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS))


def mask_of(p, size=512):
    return (np.asarray(Image.open(p).convert("L").resize((size, size), Image.NEAREST)) > 25).astype(np.uint8)


fold = "test" if a.heldout else "train"
for cat in cats:
    sp = os.path.join(a.cat_root, cat, "split_0")
    prompts = json.load(open(os.path.join(sp, f"{fold}_prompts.json"), encoding="utf-8"))
    # ---- compositing of the generated pools
    for var in ("cattext", "catcolor"):
        src = os.path.join(a.pools, f"{var}_heldout_{cat}" if a.heldout else f"{var}_syn_{cat}")
        dst = os.path.join(a.out, f"{var}_heldout_{cat}" if a.heldout else f"comp_{var}_{cat}")
        if not os.path.isdir(src):
            print("missing pool", src); continue
        os.makedirs(dst, exist_ok=True)
        for f in ("prompts.json", "labels.json"):
            if os.path.exists(os.path.join(src, f)):
                open(os.path.join(dst, f), "w", encoding="utf-8").write(open(os.path.join(src, f), encoding="utf-8").read())
        names = json.load(open(os.path.join(src, "prompts.json"), encoding="utf-8")) if os.path.exists(os.path.join(src, "prompts.json")) \
            else [n for n in os.listdir(src) if n.endswith(".png")]
        for n in names:
            g = load_rgb(os.path.join(src, n)); c = load_rgb(os.path.join(sp, f"{fold}_C", n))
            m = cv2.dilate(mask_of(os.path.join(sp, f"{fold}_A", n)), K)[..., None]
            Image.fromarray(np.where(m > 0, g, c).astype(np.uint8)).save(os.path.join(dst, n))
    if a.heldout:
        print(cat, "held-out composited"); continue
    # ---- copy-paste baseline
    dst = os.path.join(a.out, f"paste_{cat}"); os.makedirs(dst, exist_ok=True)
    labels, used = {}, {}
    for n, pmt in prompts.items():
        c = load_rgb(os.path.join(sp, "train_C", n)); b = load_rgb(os.path.join(sp, "train_B", n))
        m = mask_of(os.path.join(sp, "train_A", n)) * 255
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        try:
            out = cv2.seamlessClone(b, c, m, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:  # mask touching the border: plain paste
            mm = (m > 0)[..., None]; out = np.where(mm, b, c)
        Image.fromarray(out.astype(np.uint8)).save(os.path.join(dst, n))
        t = parse(n)[1]
        labels[n] = {"requested": t, "mask": t, "category": cat}; used[n] = pmt
    json.dump(labels, open(os.path.join(dst, "labels.json"), "w", encoding="utf-8"), indent=1)
    json.dump(used, open(os.path.join(dst, "prompts.json"), "w", encoding="utf-8"), indent=1)
    print(cat, "done:", len(used), "images per pool")
