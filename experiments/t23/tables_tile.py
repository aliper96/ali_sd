"""
tables_tile.py — LaTeX table bodies for the tile study, generated from the result files (no hand-copying).

Inputs (downloaded from the A100 to --live): results/ (real-only D_S, D_S_AUG), results_textfix/ and
results_colorfix/ (CAS, NAS), results_judge/{textfix,colorfix}_{heldout,swap}_split*_seed*.json.
Outputs in --out:
  tab_tile_binary.tex, tab_tile_multiclass.tex : rows metric x {D_S, D_S_AUG, CAS text, CAS colour,
      NAS text, NAS colour}; cells mean +/- std over split x seed runs; best mean in bold; the p-value of
      a paired Wilcoxon test vs D_S_AUG (pairs = split x seed) is given for the NAS columns.
  tab_tile_judge.tex : label fidelity per carrier and mode (mean +/- std over split x judge seed).
  tab_tile_numbers.json : every number used, for the text.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for var, sub in (("real", "results"), ("text", "results_textfix"), ("colour", "results_colorfix")):
    for f in glob.glob(os.path.join(a.live, sub, "split_*", "*", "*.csv")):
        if f.endswith("_summary.csv"):
            continue
        p = f.replace("\\", "/").split("/")
        df = pd.read_csv(f)
        df["variant"], df["split"], df["task"] = var, int(p[-3][6:]), p[-2]
        rows.append(df)
d = pd.concat(rows)
COLS = [("real", "D_S"), ("real", "D_S_AUG"), ("text", "CAS"), ("colour", "CAS"), ("text", "NAS"), ("colour", "NAS")]
METRICS = {"binary": [("Precision (B)", "precision"), ("Recall (B)", "recall"),
                      ("mAP$_{50}$ (B)", "mAP50"), ("mAP$_{50\\text{--}95}$ (B)", "mAP50_95")],
           "multiclass": [("Precision (M)", "seg_precision"), ("Recall (M)", "seg_recall"),
                          ("mAP$_{50}$ (M)", "seg_mAP50"), ("mAP$_{50\\text{--}95}$ (M)", "seg_mAP50_95")]}
numbers = {"n_runs": {}}
os.makedirs(a.out, exist_ok=True)


def fmt_p(p):
    return "p$<$0.001" if p < 0.001 else f"p={p:.3f}"


for task, metrics in METRICS.items():
    t = d[d.task == task]
    lines = []
    for label, col in metrics:
        series = {(v, P): t[(t.variant == v) & (t.protocol == P)].set_index(["split", "seed"])[col]
                  for v, P in COLS}
        best = max(series, key=lambda k: series[k].mean())
        ref = series[("real", "D_S_AUG")]
        cells = []
        for key in COLS:
            s = series[key]
            txt = f"{s.mean():.3f} $\\pm$ {s.std():.3f}"
            if key == best:
                txt = f"\\textbf{{{s.mean():.3f}}} $\\pm$ {s.std():.3f}"
            if key[1] == "NAS":
                i = s.index.intersection(ref.index)
                pv = wilcoxon(s[i], ref[i]).pvalue
                txt += f" \\newline {{\\scriptsize {fmt_p(pv)}}}"
                numbers[f"{task}/{col}/{key[0]}_NAS_p"] = pv
            cells.append(txt)
            numbers[f"{task}/{col}/{key[0]}_{key[1]}"] = [float(s.mean()), float(s.std())]
            numbers["n_runs"][f"{task}/{key[0]}_{key[1]}"] = int(len(s))
        lines.append(f"        {label} & " + " & ".join(cells) + " \\\\")
    for P in ("CAS", "NAS"):  # text vs colour
        col = metrics[3][1]
        x = t[(t.variant == "text") & (t.protocol == P)].set_index(["split", "seed"])[col]
        y = t[(t.variant == "colour") & (t.protocol == P)].set_index(["split", "seed"])[col]
        i = x.index.intersection(y.index)
        numbers[f"{task}/{col}/text_vs_colour_{P}_p"] = float(wilcoxon(x[i], y[i]).pvalue)
    # complete tabular: \input cannot be used between the rows of an alignment (it is robust in LaTeX 2020+)
    head = (r"\begin{tabular}{@{}l *{6}{>{\centering\arraybackslash}m{0.125\textwidth}}@{}}" "\n"
            r"\toprule" "\n"
            r"& \textbf{D\_S} & \textbf{D\_S\_\allowbreak AUG} & \textbf{CAS text} & \textbf{CAS colour} & "
            r"\textbf{NAS text} & \textbf{NAS colour} \\ \midrule" "\n")
    open(os.path.join(a.out, f"tab_tile_{task}.tex"), "w", encoding="utf-8").write(
        head + " \\hdashline\n".join(lines) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

# judge
J = {}
for f in glob.glob(os.path.join(a.live, "results_judge", "*fix_*.json")):
    b = os.path.basename(f)
    var = {"textfix": "text", "colorfix": "colour"}[b.split("_")[0]]
    mode = b.split("_")[1]
    k, s = int(b.split("_split")[1][0]), int(b.split("seed")[1][0])
    J.setdefault((var, mode), {})[(k, s)] = json.load(open(f))
jl = []
for var in ("text", "colour"):
    h, w = J[(var, "heldout")], J[(var, "swap")]
    cell = lambda dd, key: f"{np.mean([x[key] for x in dd.values()]):.3f} $\\pm$ {np.std([x[key] for x in dd.values()], ddof=1):.3f}"  # noqa: E731
    jl.append(f"        {'Text prompt' if var == 'text' else 'Colour-coded mask'} & {cell(h, 'ceiling')} & "
              f"{cell(h, 'fidelity')} & {cell(w, 'fidelity')} & {cell(w, 'follows_mask')} \\\\")
    for mode, dd in (("heldout", h), ("swap", w)):
        for key in ("ceiling", "fidelity", "follows_mask", "chance"):
            numbers[f"judge/{var}/{mode}/{key}"] = float(np.mean([x[key] for x in dd.values()]))
        numbers[f"judge/{var}/{mode}/n"] = len(dd)
for mode in ("heldout", "swap"):
    c = sorted(set(J[("text", mode)]) & set(J[("colour", mode)]))
    numbers[f"judge/text_vs_colour/{mode}/p"] = float(wilcoxon([J[("text", mode)][i]["fidelity"] for i in c],
                                                               [J[("colour", mode)][i]["fidelity"] for i in c]).pvalue)
    numbers[f"judge/text_vs_colour/{mode}/n"] = len(c)
open(os.path.join(a.out, "tab_tile_judge.tex"), "w", encoding="utf-8").write(
    r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
    r"\textbf{Class carrier} & \textbf{Ceiling} & \textbf{Fidelity} & \textbf{Swap: requested} & "
    r"\textbf{Swap: shape} \\" "\n" r"\midrule" "\n"
    + "\n".join(jl) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
json.dump(numbers, open(os.path.join(a.out, "tab_tile_numbers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in numbers.items() if "p" in k.split("/")[-1] or k.startswith("judge")}, indent=1))
