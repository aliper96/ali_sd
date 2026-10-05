"""
sam_analysis.py — analysis of the SAM step (PLAN.md, SAM 3 study, amendment 2) from sam_box.py outputs and the judge.

Per category: SAM-vs-ground-truth IoU on real test defects (reference); for the held-out synthetic pools (text, colour,
AD): IoU with the requested mask and acceptance; with the per-image judge verdicts (3 seeds, majority vote), the label
fidelity of all images vs accepted images, the share of judge-wrong images rejected (errors caught) and of judge-right
images rejected (false rejections). Thresholds 0.25 (primary), 0.10 and 0.50 (sensitivity) are applied to the stored
IoUs. Wilcoxon over categories for accepted vs all fidelity.

    python sam_analysis.py --live C:/Users/aliha/Desktop/PHD/t23_runs/live
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
a = ap.parse_args()
B = os.path.join(a.live, "results_sam_box")
CATS = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper".split()
SRC = {"text": "cattext", "colour": "catcolor", "ad": "ad"}


def recs(name):
    f = os.path.join(B, name + ".json")
    return {r["name"]: r for r in json.load(open(f))["records"]} if os.path.exists(f) else None


def judge_ok(var, cat):
    """name -> True if the majority of judge seeds assign the requested type."""
    fs = sorted(glob.glob(os.path.join(a.live, "results_judge", f"{var}_heldout_{cat}_seed*.json")))
    if not fs:
        return None
    votes = {}
    for f in fs:
        d = json.load(open(f))
        votes[f] = {x["name"] for x in d["disagreements"]}  # names judged as another type
    return votes


out = {"real": {}, "pools": {}, "train": {}}
for c in CATS:
    r = recs(f"real_{c}")
    if r:
        out["real"][c] = {"iou": float(np.mean([x["iou"] for x in r.values()])), "n": len(r),
                          "acc": float(np.mean([x["accepted"] for x in r.values()]))}
    for s in ("text", "colour"):
        t = recs(f"train_{s}_{c}")
        if t:
            out["train"].setdefault(s, {})[c] = {"acc": float(np.mean([x["accepted"] for x in t.values()])),
                                                 "iou": float(np.mean([x["iou"] for x in t.values()])), "n": len(t)}
    for s, var in SRC.items():
        p = recs(f"heldout_{s}_{c}")
        if not p:
            continue
        row = {"iou": float(np.mean([x["iou"] for x in p.values()])), "n": len(p)}
        v = judge_ok(var, c) if c != "toothbrush" else None
        if v:
            names = sorted(p)
            ok = {n: np.mean([n not in w for w in v.values()]) >= 0.5 for n in names}  # majority of seeds right
            for thr in (0.10, 0.25, 0.50):
                acc = {n: p[n]["iou"] >= thr for n in names}
                A = [n for n in names if acc[n]]
                wrong = [n for n in names if not ok[n]]
                right = [n for n in names if ok[n]]
                row[f"t{thr:.2f}"] = {
                    "acc_rate": float(np.mean(list(acc.values()))),
                    "fid_all": float(np.mean([ok[n] for n in names])),
                    "fid_acc": float(np.mean([ok[n] for n in A])) if A else float("nan"),
                    "wrong_rejected": float(np.mean([not acc[n] for n in wrong])) if wrong else float("nan"),
                    "right_rejected": float(np.mean([not acc[n] for n in right])) if right else float("nan"),
                    "n_wrong": len(wrong)}
        out["pools"].setdefault(s, {})[c] = row

summ = {"real_iou": float(np.mean([v["iou"] for v in out["real"].values()])) if out["real"] else None,
        "n_real_cat": len(out["real"])}
for s in SRC:
    P = out["pools"].get(s, {})
    summ[f"{s}_iou"] = float(np.mean([v["iou"] for v in P.values()])) if P else None
    for thr in ("t0.10", "t0.25", "t0.50"):
        J = {c: v[thr] for c, v in P.items() if thr in v and not np.isnan(v[thr]["fid_acc"])}
        if len(J) < 3:
            continue
        fa = np.array([J[c]["fid_all"] for c in J]); fc = np.array([J[c]["fid_acc"] for c in J])
        summ[f"{s}_{thr}"] = {"n_cat": len(J), "acc_rate": float(np.mean([J[c]["acc_rate"] for c in J])),
                              "fid_all": float(fa.mean()), "fid_acc": float(fc.mean()),
                              "p_fid": float(wilcoxon(fc, fa).pvalue) if np.any(fc != fa) else 1.0,
                              "wrong_rejected": float(np.nanmean([J[c]["wrong_rejected"] for c in J])),
                              "right_rejected": float(np.nanmean([J[c]["right_rejected"] for c in J]))}
for s in ("text", "colour"):
    T = out["train"].get(s, {})
    if T:
        summ[f"train_{s}_acc"] = float(np.mean([v["acc"] for v in T.values()]))
out["summary"] = summ
json.dump(out, open(os.path.join(a.live, "sam_analysis.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
