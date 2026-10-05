"""
aggregate.py — Tables 2-3 from the re-run: 5 splits x 3 detector seeds = 15 runs/cell.

Reads <results>/split_<k>/<task>/<P>.csv (written by run_downstream_yolo.py) and prints,
per task (binary = Table 2 with box metrics, multiclass = Table 3 with mask metrics):
  * mean +/- std over the 15 runs and a bootstrap 95% CI of the mean (S3);
  * a paired Wilcoxon signed-rank test of CAS, NAS and D_S vs D_S_AUG, pairing runs by
    (split, seed) (S4). No claim of improvement is made without its p-value.
Writes <out>/tab_t23_<task>.tex (table body rows) and <out>/t23_summary.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from stats_utils import bootstrap_ci  # noqa: E402

PROTOCOLS = ["D_S", "D_S_AUG", "CAS", "NAS"]
METRICS = {  # table row label -> csv column
    "binary": [("precision (B)", "precision"), ("recall (B)", "recall"),
               ("mAP50 (B)", "mAP50"), ("mAP50-95 (B)", "mAP50_95")],
    "multiclass": [("precision (M)", "seg_precision"), ("recall (M)", "seg_recall"),
                   ("mAP50 (M)", "seg_mAP50"), ("mAP50-95 (M)", "seg_mAP50_95")],
}

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
res, out = Path(a.results), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

summary = []
for task, metrics in METRICS.items():
    frames = []
    for csv in sorted(res.glob(f"split_*/{task}/*.csv")):
        if csv.name.endswith("_summary.csv"):
            continue
        df = pd.read_csv(csv)
        df["split"] = int(csv.parts[-3].split("_")[1])
        frames.append(df)
    if not frames:
        print(f"[{task}] no results yet")
        continue
    df = pd.concat(frames)
    print(f"\n=== {task}: runs per protocol =", df.groupby("protocol").size().to_dict())
    rows_tex = []
    for label, col in metrics:
        cells = []
        best = max(PROTOCOLS, key=lambda p: df.loc[df.protocol == p, col].mean()
                   if (df.protocol == p).any() else -1)
        for p in PROTOCOLS:
            v = df.loc[df.protocol == p, col].to_numpy()
            if len(v) == 0:
                cells.append("n/a")
                continue
            m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            lo, hi = bootstrap_ci(v)
            txt = f"${m:.3f} \\pm {s:.3f}$"
            cells.append(f"\\textbf{{{txt}}}" if p == best else txt)
            rec = dict(task=task, metric=col, protocol=p, n=len(v), mean=m, std=s,
                       ci_low=lo, ci_high=hi)
            if p != "D_S_AUG":
                ref = df[df.protocol == "D_S_AUG"].set_index(["split", "seed"])[col]
                cur = df[df.protocol == p].set_index(["split", "seed"])[col]
                idx = ref.index.intersection(cur.index)
                if len(idx) >= 5 and not np.allclose(cur[idx].values, ref[idx].values):
                    rec["wilcoxon_p_vs_D_S_AUG"] = float(wilcoxon(cur[idx], ref[idx]).pvalue)
                    rec["n_pairs"] = len(idx)
            summary.append(rec)
            print(f"  {label:14s} {p:8s} n={len(v):2d}  {m:.3f} ± {s:.3f}  CI[{lo:.3f},{hi:.3f}]"
                  + (f"  p={rec['wilcoxon_p_vs_D_S_AUG']:.4f}" if "wilcoxon_p_vs_D_S_AUG" in rec else ""))
        rows_tex.append(f"        {label} & " + " & ".join(cells) + " \\\\")
    (out / f"tab_t23_{task}.tex").write_text(" \\hdashline\n".join(rows_tex) + "\n", encoding="utf-8")

pd.DataFrame(summary).to_csv(out / "t23_summary.csv", index=False)
print("\nwrote", out / "t23_summary.csv")
