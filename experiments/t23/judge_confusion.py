"""judge_confusion.py — confusion matrices of the label-fidelity judge (review 2026-10-05).
Rows = requested type, columns = type assigned by the judge; counts pooled over splits and judge seeds.
Built from results_judge/*.json: per_class gives n_syn per requested type (tile) or per_category (categories),
`disagreements` lists every image the judge did not assign to the requested type; the diagonal is the rest.
  python prep/t23/judge_confusion.py --live C:/Users/aliha/Desktop/PHD/t23_runs/live --out FirstPaper_renaming/generated/tab_judge_confusion.tex
"""
import argparse, glob, json, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
TILE = {"textfix": "Text prompt", "colorfix": "Colour-coded mask"}
L = []


def matrix(files, cat=None):
    """returns (types, {req: {jud: n}})"""
    M = defaultdict(lambda: defaultdict(int)); types = set()
    for f in files:
        d = json.load(open(f))
        if cat is None and "per_class" not in d:
            continue
        if cat is None:
            pc = d["per_class"]; n_of = {t: v["n"] if isinstance(v, dict) else v for t, v in pc.items()}
        else:
            n_of = {}
            for dd in d["disagreements"]:
                pass
            # categories: n_syn per requested type is not stored; count from disagreements + labels of the pool
            pool = d["syn"]; lab = os.path.join(a.live, "..", "..")  # not available locally -> use per_category n_syn / types
            pcat = d["per_category"][cat]
            per = pcat["n_syn"] / len(pcat["types"])
            n_of = {t: None for t in pcat["types"]}
        dis = [x for x in d["disagreements"] if cat is None or x.get("category") == cat]
        off = defaultdict(lambda: defaultdict(int))
        for x in dis:
            off[x["requested"]][x["judged"]] += 1
        for t, n in n_of.items():
            types.add(t)
            if n is None:
                continue
            for j, k in off[t].items():
                M[t][j] += k; types.add(j)
            M[t][t] += n - sum(off[t].values())
    return sorted(types), M


def table(title, types, M, label):
    L.append(r"\begin{table}[ht]\centering\footnotesize")
    L.append(r"\caption{" + title + r" Rows: requested type; columns: type assigned by the judge; counts pooled over splits and judge seeds.}")
    L.append(r"\label{" + label + "}")
    L.append(r"\begin{tabular}{@{}l" + "c" * len(types) + "@{}}\\toprule")
    L.append("requested $\\downarrow$ / judged & " + " & ".join(t.replace("_", "\\_") for t in types) + r" \\\midrule")
    for t in types:
        row = [str(M[t][j]) if M[t][j] else "--" for j in types]
        L.append(t.replace("_", "\\_") + " & " + " & ".join(("\\textbf{" + v + "}") if j == t and v != "--" else v for j, v in zip(types, row)) + r" \\")
    L.append(r"\bottomrule\end{tabular}\end{table}")


for var, name in TILE.items():
    for mode, mname in [("heldout", "requested class = class of the mask"), ("swap", "swap test, requested class $\\neq$ class of the mask")]:
        files = sorted(glob.glob(os.path.join(a.live, "results_judge", f"{var}_{mode}_split*_seed*.json")))
        if not files:
            continue
        types, M = matrix(files)
        table(f"\\textit{{Tile}}, {name}, {mname} ({len(files)} judge runs).", types, M, f"tab:conf_{var}_{mode}")
open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", a.out, "tables:", L.count(r"\begin{table}[ht]\centering\footnotesize"))
