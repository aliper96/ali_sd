"""
tables_cat.py — LaTeX tables and number macros for the per-category MVTec-AD study (one LoRA per category and
label carrier, split 0), generated from the downloaded result files (no hand-copying).

Inputs (--live): results_catreal/<cat>/multiclass/D_S_AUG.csv, results_cat{text,color}/<cat>/multiclass/{CAS,NAS}.csv,
results_judge/cat{text,color}_{heldout,swap}_<cat>_seed*.json, results_timing/*.json; --splits: per-category split
folders (for the train/test counts).
Outputs (--out):
  tab_cat_detector.tex : per category, mask mAP50-95 of YOLOv8n-seg (mean +/- std over 3 detector seeds) for
                         real-only (D_S_AUG), CAS and NAS with each carrier; mean row; tests.
  tab_cat_judge.tex    : judge ceiling, fidelity and swap (requested) per carrier; mean row; tests.
  tab_efficiency.tex   : generator inference cost (measured JSONs only).
  cat_numbers.tex      : \\newcommand macros with every number quoted in the text.
  cat_numbers.json     : the same numbers, for checking.
Only categories with ALL detector cells present are used; the number of categories is a macro.
Statistics: paired Wilcoxon over category x seed pairs, Wilcoxon over the per-category means (conservative),
and a 95% CI of the mean difference by bootstrap over categories (10,000 resamples, seed 0).
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
ap.add_argument("--splits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="5090", help="suffix of results_timing/*_<gpu>.json to report")
a = ap.parse_args()

CATS = "bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper".split()
TEXTURES = {"carpet", "grid", "leather", "wood"}
M = "seg_mAP50_95"


def seeds(sub, cat, proto):
    f = os.path.join(a.live, sub, cat, "multiclass", f"{proto}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    return df.set_index("seed")[M]


def judge(var, mode, cat, key="fidelity"):
    fs = sorted(glob.glob(os.path.join(a.live, "results_judge_train" if mode == "train" else "results_judge", f"{var}_{mode}_{cat}_seed*.json")))
    if not fs:
        return None
    js = [json.load(open(f)) for f in fs]
    return pd.Series({j["seed"]: j[key] for j in js})


det = {}
for c in CATS:
    cells = {"real": seeds("results_catreal", c, "D_S_AUG"),
             "cas_t": seeds("results_cattext", c, "CAS"), "cas_c": seeds("results_catcolor", c, "CAS"),
             "nas_t": seeds("results_cattext", c, "NAS"), "nas_c": seeds("results_catcolor", c, "NAS")}
    if all(v is not None and len(v) == 3 for v in cells.values()):
        det[c] = cells
cats = list(det)
jcats = []
jud = {}
for c in cats:
    if c == "toothbrush":  # one defect type: classification is undefined
        continue
    cells = {"ceil": judge("cattext", "heldout", c, "ceiling"),
             "fid_t": judge("cattext", "heldout", c), "fid_c": judge("catcolor", "heldout", c),
             "swap_t": judge("cattext", "swap", c), "swap_c": judge("catcolor", "swap", c)}
    if all(v is not None and len(v) == 3 for v in cells.values()):
        jud[c] = cells
        jcats.append(c)


def counts(c):
    sp = os.path.join(a.splits, c, "split_0")
    n = lambda f: len(json.load(open(os.path.join(sp, f"{f}_prompts.json"))))
    types = {"_".join(k[:-4].split("_")[len(c.split("_")):-1]) for k in json.load(open(os.path.join(sp, "train_prompts.json")))}
    return n("train"), n("test"), len(types)


rng = np.random.default_rng(0)


def compare(x_key, y_key, table, cs):
    """y vs x: paired over category x seed, at category level, and bootstrap CI over categories."""
    xs, ys, cl = [], [], []
    for c in cs:
        x, y = table[c][x_key], table[c][y_key]
        for s in sorted(set(x.index) & set(y.index)):
            xs.append(x[s]); ys.append(y[s]); cl.append(c)
    xs, ys, cl = np.array(xs), np.array(ys), np.array(cl)
    d = ys - xs
    per = {c: d[cl == c] for c in cs}
    boot = [np.mean(np.concatenate([per[c] for c in rng.choice(cs, len(cs))])) for _ in range(10000)]
    cm = np.array([per[c].mean() for c in cs])
    return {"n_pairs": int(len(d)), "x": float(xs.mean()), "y": float(ys.mean()), "diff": float(d.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "p_pairs": float(wilcoxon(ys, xs).pvalue), "p_cat": float(wilcoxon(cm).pvalue),
            "n_up": int((cm > 0).sum()), "n_cat": len(cs)}


N = {}
N["nas_t_vs_real"] = compare("real", "nas_t", det, cats)
N["nas_c_vs_real"] = compare("real", "nas_c", det, cats)
N["cas_t_vs_real"] = compare("real", "cas_t", det, cats)
N["cas_c_vs_real"] = compare("real", "cas_c", det, cats)
N["nas_c_vs_t"] = compare("nas_t", "nas_c", det, cats)
N["fid_c_vs_t"] = compare("fid_t", "fid_c", jud, jcats)
N["swap_c_vs_t"] = compare("swap_t", "swap_c", jud, jcats)
N["ceil_mean"] = float(np.mean([jud[c]["ceil"].mean() for c in jcats]))
N["n_cat"], N["n_jcat"] = len(cats), len(jcats)


def lab(c):
    return c.replace("_", "\\_") + ("$^\\dagger$" if c in TEXTURES else "")


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fpt(p):  # for running text: a complete math expression
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def ms(s):
    return f"{s.mean():.3f}$\\pm${s.std(ddof=1):.3f}"


os.makedirs(a.out, exist_ok=True)
# ---- detector table
cols = ["real", "cas_t", "cas_c", "nas_t", "nas_c"]
L = ["\\begin{tabular}{@{}l c c c c c c c@{}}", "\\toprule",
     "Category & $n_\\text{tr}/n_\\text{te}$ & Types & Real only (D\\_S\\_AUG) & CAS text & CAS colour & NAS text & NAS colour \\\\",
     "\\midrule"]
for c in cats:
    ntr, nte, nty = counts(c)
    best = max(cols, key=lambda k: det[c][k].mean())
    cells = [("\\textbf{%s}" % ms(det[c][k])) if k == best else ms(det[c][k]) for k in cols]
    L.append(f"{lab(c)} & {ntr}/{nte} & {nty} & " + " & ".join(cells) + " \\\\")
L.append("\\midrule")
means = {k: np.mean([det[c][k].mean() for c in cats]) for k in cols}
L.append(f"Mean ({len(cats)} categories) & & & " + " & ".join(f"{means[k]:.3f}" for k in cols) + " \\\\")
for k, key in (("cas_t", "cas_t_vs_real"), ("cas_c", "cas_c_vs_real"), ("nas_t", "nas_t_vs_real"), ("nas_c", "nas_c_vs_real")):
    pass
L.append("$p$ vs.\\ real only (categories / pairs) & & & -- & "
         + " & ".join(f"{fp(N[k]['p_cat'])} / {fp(N[k]['p_pairs'])}" for k in ("cas_t_vs_real", "cas_c_vs_real", "nas_t_vs_real", "nas_c_vs_real")) + " \\\\")
L.append("Categories above real only & & & -- & "
         + " & ".join(f"{N[k]['n_up']}/{N[k]['n_cat']}" for k in ("cas_t_vs_real", "cas_c_vs_real", "nas_t_vs_real", "nas_c_vs_real")) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(a.out, "tab_cat_detector.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- judge table
jcols = ["ceil", "fid_t", "fid_c", "swap_t", "swap_c"]
L = ["\\begin{tabular}{@{}l c c c c c c@{}}", "\\toprule",
     "Category & Chance & Ceiling & Fidelity text & Fidelity colour & Swap text & Swap colour \\\\", "\\midrule"]
for c in jcats:
    nty = counts(c)[2]
    row = [f"{jud[c][k].mean():.2f}" for k in jcols]
    for pair in (("fid_t", "fid_c"), ("swap_t", "swap_c")):
        i, j = jcols.index(pair[0]), jcols.index(pair[1])
        if row[i] == row[j]:  # tie at the printed precision: bold both
            row[i], row[j] = "\\textbf{%s}" % row[i], "\\textbf{%s}" % row[j]
        else:
            b = i if jud[c][pair[0]].mean() > jud[c][pair[1]].mean() else j
            row[b] = "\\textbf{%s}" % row[b]
    L.append(f"{lab(c)} & {1 / nty:.2f} & " + " & ".join(row) + " \\\\")
L.append("\\midrule")
jm = {k: np.mean([jud[c][k].mean() for c in jcats]) for k in jcols}
N["chance_mean"] = float(np.mean([1 / counts(c)[2] for c in jcats]))  # average uniform-chance rate of the judge categories
L.append(f"Mean ({len(jcats)} categories) & & " + " & ".join(f"{jm[k]:.3f}" for k in jcols) + " \\\\")
L.append(f"$p$ colour vs.\\ text (categories / pairs) & & & \\multicolumn{{2}}{{c}}{{{fp(N['fid_c_vs_t']['p_cat'])} / {fp(N['fid_c_vs_t']['p_pairs'])}}} & "
         f"\\multicolumn{{2}}{{c}}{{{fp(N['swap_c_vs_t']['p_cat'])} / {fp(N['swap_c_vs_t']['p_pairs'])}}} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(a.out, "tab_cat_judge.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- efficiency table (measured only)
L = ["\\begin{tabular}{@{}l c c c c c c l@{}}", "\\toprule",
     "Generator & Steps & Resolution & Batch & s / image & Images / s & Peak memory & Trained parameters \\\\",
     "\\midrule"]
tim = {}
mac_ad50 = None
TGPU = a.gpu
for f in sorted(glob.glob(os.path.join(a.live, "results_timing", f"*_{TGPU}.json"))):
    t = json.load(open(f))
    if os.path.basename(f).startswith("ad50"):  # AD at 50 DDIM steps (review 2026-10-05): macro only
        mac_ad50 = t["s_per_img_mean"]
        continue
    ours = t["method"].startswith("Ali-AUG")
    name = "Ali-AUG (ours)" if ours else "AnomalyDiffusion~\\cite{hu2024anomalydiffusionfewshotanomalyimage}"
    scope = "per category" if ours else "one model, all categories"
    tim[os.path.basename(f)[:-5]] = t
    L.append(f"{name} & {t['steps']} & {t['resolution']}$^2$ & {t['batch']} & {t['s_per_img_mean']:.3f} & "
             f"{t['img_per_s']:.2f} & {t['peak_mem_gb']:.1f}\\,GB & {t['trained_params_M']:.1f}\\,M ({scope}) \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(a.out, "tab_efficiency.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---- AnomalyDiffusion comparison (released checkpoint; results_ad, results_clean)
ovl_f = os.path.join(a.live, "ad_overlap_split0.json")
AD = {}
if os.path.exists(ovl_f) and os.path.isdir(os.path.join(a.live, "results_ad")):
    ovl = json.load(open(ovl_f))
    full = {c: {"real": det[c]["real"], "ad": seeds("results_ad", c, "NAS"), "text": det[c]["nas_t"],
                "colour": det[c]["nas_c"], "ad_cas": seeds("results_ad", c, "CAS")} for c in cats}
    acats = [c for c in cats if full[c]["ad"] is not None and len(full[c]["ad"]) == 3]
    clean = {}
    for c in acats:
        cl = {"real": seeds("results_clean/real", c, "D_S_AUG"), "ad": seeds("results_clean/ad", c, "NAS"),
              "text": seeds("results_clean/cattext", c, "NAS"), "colour": seeds("results_clean/catcolor", c, "NAS")}
        if all(v is not None and len(v) == 3 for v in cl.values()):
            clean[c] = cl
    ccats = list(clean)
    adj = {c: judge("ad", "heldout", c) for c in jcats}
    ajcats = [c for c in jcats if adj[c] is not None and len(adj[c]) == 3]
    for c in ajcats:
        jud[c]["fid_ad"] = adj[c]
    AD["full_c_vs_ad"] = compare("ad", "colour", full, acats)
    AD["full_t_vs_ad"] = compare("ad", "text", full, acats)
    AD["full_ad_vs_real"] = compare("real", "ad", full, acats)
    AD["cas_c_vs_ad"] = compare("ad_cas", "colour", {c: {"ad_cas": full[c]["ad_cas"], "colour": det[c]["cas_c"]} for c in acats}, acats)
    AD["cas_t_vs_ad"] = compare("ad_cas", "text", {c: {"ad_cas": full[c]["ad_cas"], "text": det[c]["cas_t"]} for c in acats}, acats)
    AD["fid_c_vs_ad"] = compare("fid_ad", "fid_c", jud, ajcats)
    AD["fid_t_vs_ad"] = compare("fid_ad", "fid_t", jud, ajcats)
    if len(ccats) >= 5:
        AD["clean_c_vs_ad"] = compare("ad", "colour", clean, ccats)
        AD["clean_t_vs_ad"] = compare("ad", "text", clean, ccats)
        AD["clean_ad_vs_real"] = compare("real", "ad", clean, ccats)
        AD["clean_c_vs_real"] = compare("real", "colour", clean, ccats)
        AD["clean_t_vs_real"] = compare("real", "text", clean, ccats)
    seen_t = sum(ovl[c]["test_seen_by_AD"] for c in acats)
    tot_t = sum(ovl[c]["test_total"] for c in acats)
    AD["seen"], AD["test_total"], AD["n_cat"], AD["n_clean"], AD["n_jcat"] = seen_t, tot_t, len(acats), len(ccats), len(ajcats)
    # per-category detector table: full test and unseen test
    L = ["\\begin{tabular}{@{}l c cccc c cccc@{}}", "\\toprule",
         " & & \\multicolumn{4}{c}{Full test fold} & & \\multicolumn{4}{c}{Test images unseen by AD} \\\\",
         "\\cmidrule(lr){3-6}\\cmidrule(lr){8-11}",
         "Category & Seen by AD & Real only & AD & Ali-AUG text & Ali-AUG colour & & Real only & AD & Ali-AUG text & Ali-AUG colour \\\\",
         "\\midrule"]
    keys = ["real", "ad", "text", "colour"]
    for c in acats:
        fm = {k: full[c][k].mean() for k in keys}
        bf = max(keys, key=lambda k: fm[k])
        cells = ["\\textbf{%.3f}" % fm[k] if k == bf else "%.3f" % fm[k] for k in keys]
        if c in clean:
            cm = {k: clean[c][k].mean() for k in keys}
            bc = max(keys, key=lambda k: cm[k])
            cc = ["\\textbf{%.3f}" % cm[k] if k == bc else "%.3f" % cm[k] for k in keys]
            nun = ovl[c]["test_total"] - ovl[c]["test_seen_by_AD"]
        else:
            cc, nun = ["--"] * 4, None
        L.append(f"{lab(c)} & {ovl[c]['test_seen_by_AD']}/{ovl[c]['test_total']} & " + " & ".join(cells) + " & & " + " & ".join(cc) + " \\\\")
    L.append("\\midrule")
    L.append(f"Mean ({len(acats)} / {len(ccats)} categories) & {seen_t}/{tot_t} & "
             + " & ".join("%.3f" % np.mean([full[c][k].mean() for c in acats]) for k in keys) + " & & "
             + " & ".join("%.3f" % np.mean([clean[c][k].mean() for c in ccats]) for k in keys) + " \\\\")
    def prow(name, kf, kc):
        f, g = AD[kf], AD.get(kc)
        return (f"{name} & & \\multicolumn{{4}}{{c}}{{{fp(f['p_cat'])} / {fp(f['p_pairs'])} ({f['n_up']}/{f['n_cat']})}} & & "
                + (f"\\multicolumn{{4}}{{c}}{{{fp(g['p_cat'])} / {fp(g['p_pairs'])} ({g['n_up']}/{g['n_cat']})}}" if g else "\\multicolumn{4}{c}{--}") + " \\\\")
    L.append(prow("$p$ colour vs.\\ AD", "full_c_vs_ad", "clean_c_vs_ad"))
    L.append(prow("$p$ text vs.\\ AD", "full_t_vs_ad", "clean_t_vs_ad"))
    L.append(prow("$p$ AD vs.\\ real only", "full_ad_vs_real", "clean_ad_vs_real"))
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_ad_detector.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    # quality table: label fidelity, CAS, background preservation, sharpness
    pl = json.load(open(os.path.join(a.live, "preserve_lpips.json")))
    pm = json.load(open(os.path.join(a.live, "preserve_metrics.json")))["256"]
    tex = [c for c in pm["per_category"] if c in TEXTURES]
    obj = [c for c in pm["per_category"] if c not in TEXTURES]
    sharp = lambda m, cs: np.mean([pm["per_category"][c][m]["sharp"] for c in cs])
    meth = [("AnomalyDiffusion", "fid_ad", "ad_cas", "AnomalyDiffusion"), ("Ali-AUG text", "fid_t", "cas_t", "Ali-AUG text"),
            ("Ali-AUG colour", "fid_c", "cas_c", "Ali-AUG colour")]
    icf = os.path.join(a.live, "ic_lpips.json")
    IC = json.load(open(icf)) if os.path.exists(icf) else None
    icm = lambda pk: np.mean([IC[c][pk] for c in IC if IC[c].get(pk) is not None]) if IC else None
    cpl = os.path.join(a.live, "preserve_lpips_comp.json"); cpm = os.path.join(a.live, "preserve_metrics_comp.json")
    COMP = (json.load(open(cpl)), json.load(open(cpm))["256"]) if os.path.exists(cpl) and os.path.exists(cpm) else None
    L = ["\\begin{tabular}{@{}l c c c c c c c@{}}", "\\toprule",
         " & Label fidelity & CAS & IC-LPIPS & \\multicolumn{2}{c}{LPIPS outside mask $\\downarrow$} & \\multicolumn{2}{c}{Sharpness outside mask} \\\\",
         "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Generator & ($%d$ cat.) & ($%d$ cat.) & (held-out) & objects & textures & objects & textures \\\\" % (len(ajcats), len(acats)),
         "\\midrule"]
    for name, fk, ck, pk in meth:
        fv = np.mean([jud[c][fk].mean() for c in ajcats])
        cv = np.mean([(full[c]["ad_cas"] if ck == "ad_cas" else det[c][ck]).mean() for c in acats])
        ic = f"{icm(pk):.3f}" if IC else "--"
        L.append(f"{name} & {fv:.3f} & {cv:.3f} & {ic} & {pl['groups']['objects'][pk]:.3f} & {pl['groups']['textures'][pk]:.3f} & "
                 f"{sharp(pk, obj):.2f} & {sharp(pk, tex):.2f} \\\\")
    if COMP:
        cl, cm = COMP
        csharp = lambda m, cs: np.mean([cm["per_category"][c][m]["sharp"] for c in cs if c in cm["per_category"]])
        for name, pk in (("Ali-AUG text + compositing", "Ali-AUG text"), ("Ali-AUG colour + compositing", "Ali-AUG colour")):
            L.append(f"{name} & -- & -- & -- & {cl['groups']['objects'][pk]:.3f} & {cl['groups']['textures'][pk]:.3f} & "
                     f"{csharp(pk, obj):.2f} & {csharp(pk, tex):.2f} \\\\")
        AD["comp_lpips"] = cl["groups"]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_ad_quality.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    if IC:
        AD["ic_lpips"] = {pk: float(icm(pk)) for *_, pk in meth}
    AD["n_obj"], AD["n_tex"] = len(obj), len(tex)
    AD["lpips"] = pl["groups"]
    AD["sharp"] = {pk: {"objects": float(sharp(pk, obj)), "textures": float(sharp(pk, tex))} for *_, pk in meth}
    AD["means_full"] = {k: float(np.mean([full[c][k].mean() for c in acats])) for k in keys}
    AD["means_clean"] = {k: float(np.mean([clean[c][k].mean() for c in ccats])) for k in keys} if ccats else {}
    AD["fid"] = {fk: float(np.mean([jud[c][fk].mean() for c in ajcats])) for _, fk, _, _ in meth}
    AD["cas"] = {ck: float(np.mean([(full[c]["ad_cas"] if ck == "ad_cas" else det[c][ck]).mean() for c in acats])) for _, _, ck, _ in meth}
    N["ad"] = AD

# ---- synthetic:real ratio sweep (PLAN.md; ratio 1 = the per-category NAS run)
RATIO = {}
def seeds_ratio(sub, cat, ratio):
    f = os.path.join(a.live, sub, cat, "multiclass", "NAS.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df = df[np.isclose(df["ratio"].astype(float), ratio)]
    return df.set_index("seed")[M] if len(df) else None
if os.path.isdir(os.path.join(a.live, "results_catratio_catcolor")):
    rat = {}
    for c in cats:
        cell = {"real": det[c]["real"], "t1": det[c]["nas_t"], "c1": det[c]["nas_c"]}
        for r, tag in ((0.5, "05"), (2.0, "2"), (5.0, "5")):
            cell["t" + tag] = seeds_ratio("results_catratio_cattext", c, r)
            cell["c" + tag] = seeds_ratio("results_catratio_catcolor", c, r)
        if all(v is not None and len(v) == 3 for v in cell.values()):
            rat[c] = cell
    rcats = list(rat)
    if len(rcats) >= 5:
        tags = ["05", "1", "2", "5"]
        L = ["\\begin{tabular}{@{}l c cccc c cccc@{}}", "\\toprule",
             " & & \\multicolumn{4}{c}{Text carrier} & & \\multicolumn{4}{c}{Colour carrier} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(lr){8-11}",
             "Synthetic per real image & Real only & 0.5 & 1 & 2 & 5 & & 0.5 & 1 & 2 & 5 \\\\", "\\midrule"]
        mean = lambda k: np.mean([rat[c][k].mean() for c in rcats])
        L.append(f"Mean mask mAP$_{{50\\text{{--}}95}}$ ({len(rcats)} categories) & {mean('real'):.3f} & "
                 + " & ".join(f"{mean('t' + t):.3f}" for t in tags) + " & & " + " & ".join(f"{mean('c' + t):.3f}" for t in tags) + " \\\\")
        for t in tags:
            RATIO[f"t{t}_real"] = compare("real", "t" + t, rat, rcats)
            RATIO[f"c{t}_real"] = compare("real", "c" + t, rat, rcats)
        for t in ("05", "2", "5"):
            RATIO[f"t{t}_t1"] = compare("t1", "t" + t, rat, rcats)
            RATIO[f"c{t}_c1"] = compare("c1", "c" + t, rat, rcats)
        RATIO["c5_c2"] = compare("c2", "c5", rat, rcats)
        RATIO["t5_t2"] = compare("t2", "t5", rat, rcats)
        L.append("$p$ vs.\\ real only (categories) & -- & " + " & ".join(fp(RATIO[f't{t}_real']['p_cat']) for t in tags)
                 + " & & " + " & ".join(fp(RATIO[f'c{t}_real']['p_cat']) for t in tags) + " \\\\")
        L.append("$p$ vs.\\ ratio 1 (categories) & -- & " + " & ".join("--" if t == "1" else fp(RATIO[f't{t}_t1']['p_cat']) for t in tags)
                 + " & & " + " & ".join("--" if t == "1" else fp(RATIO[f'c{t}_c1']['p_cat']) for t in tags) + " \\\\")
        L.append("Categories above real only & -- & " + " & ".join(RATIO[f't{t}_real']['n_up'].__str__() + f"/{len(rcats)}" for t in tags)
                 + " & & " + " & ".join(RATIO[f'c{t}_real']['n_up'].__str__() + f"/{len(rcats)}" for t in tags) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(a.out, "tab_ratio.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        RATIO["n_cat"] = len(rcats)
        RATIO["means"] = {k: float(mean(k)) for k in ["real"] + ["t" + t for t in tags] + ["c" + t for t in tags]}
        N["ratio"] = RATIO

# ---- SAM step (contour refinement; PLAN.md SAM 3 study, amendment 2)
SB = os.path.join(a.live, "results_sam_box")
SAMN = {}
if os.path.isdir(SB) and os.path.isdir(os.path.join(a.live, "results_samstep_cattext")):
    def sbox(name):
        f = os.path.join(SB, name + ".json")
        return json.load(open(f))["records"] if os.path.exists(f) else None
    sam = {}
    for c in cats:
        st_t, st_c = seeds("results_samstep_cattext", c, "NAS"), seeds("results_samstep_catcolor", c, "NAS")
        rr, tt, cc = sbox(f"real_{c}"), sbox(f"train_text_{c}"), sbox(f"train_colour_{c}")
        if st_t is None or st_c is None or len(st_t) < 3 or len(st_c) < 3 or not (rr and tt and cc):
            continue
        sam[c] = {"real": det[c]["real"], "nas_t": det[c]["nas_t"], "nas_c": det[c]["nas_c"], "sam_t": st_t, "sam_c": st_c,
                  "iou_real": np.mean([r["iou"] for r in rr]), "acc_t": np.mean([r["accepted"] for r in tt]),
                  "acc_c": np.mean([r["accepted"] for r in cc])}
    scats = list(sam)
    if len(scats) >= 5:
        SAMN["t"] = compare("nas_t", "sam_t", sam, scats)
        SAMN["c"] = compare("nas_c", "sam_c", sam, scats)
        SAMN["t_real"] = compare("real", "sam_t", sam, scats)
        SAMN["c_real"] = compare("real", "sam_c", sam, scats)
        L = ["\\begin{tabular}{@{}l c c c c c c c@{}}", "\\toprule",
             " & SAM IoU & \\multicolumn{2}{c}{Accepted by SAM} & \\multicolumn{2}{c}{NAS text} & \\multicolumn{2}{c}{NAS colour} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
             "Category & real defects & text & colour & requested & SAM & requested & SAM \\\\", "\\midrule"]
        for c in scats:
            s = sam[c]
            def pair(x, y):
                xm, ym = s[x].mean(), s[y].mean()
                return (f"\\textbf{{{xm:.3f}}} & {ym:.3f}" if xm > ym else f"{xm:.3f} & \\textbf{{{ym:.3f}}}" if ym > xm
                        else f"{xm:.3f} & {ym:.3f}")
            L.append(f"{lab(c)} & {s['iou_real']:.2f} & {s['acc_t']:.2f} & {s['acc_c']:.2f} & {pair('nas_t', 'sam_t')} & {pair('nas_c', 'sam_c')} \\\\")
        L.append("\\midrule")
        mean = lambda k: np.mean([sam[c][k].mean() for c in scats]) if k.startswith(("nas", "sam", "real")) else np.mean([sam[c][k] for c in scats])
        L.append(f"Mean ({len(scats)} categories) & {mean('iou_real'):.2f} & {mean('acc_t'):.2f} & {mean('acc_c'):.2f} & "
                 f"{mean('nas_t'):.3f} & {mean('sam_t'):.3f} & {mean('nas_c'):.3f} & {mean('sam_c'):.3f} \\\\")
        L.append(f"$p$ SAM vs.\\ requested (categories / pairs) & & & & \\multicolumn{{2}}{{c}}{{{fp(SAMN['t']['p_cat'])} / {fp(SAMN['t']['p_pairs'])}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{fp(SAMN['c']['p_cat'])} / {fp(SAMN['c']['p_pairs'])}}} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(a.out, "tab_sam.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        SAMN["n_cat"] = len(scats)
        SAMN["means"] = {k: float(mean(k)) for k in ("iou_real", "acc_t", "acc_c", "nas_t", "sam_t", "nas_c", "sam_c", "real")}
        sa = os.path.join(a.live, "sam_analysis.json")
        if os.path.exists(sa):
            SAMN["analysis"] = json.load(open(sa))["summary"]
        N["sam"] = SAMN

# ---- fidelity of the synthetic sets actually used for CAS/NAS (training masks; review 2026-10-05)
TRN = {}
tr_tab = {}
for c in jcats:
    cells = {"train_t": judge("cattext", "train", c), "train_c": judge("catcolor", "train", c),
             "fid_t": jud[c]["fid_t"], "fid_c": jud[c]["fid_c"]}
    if all(v is not None and len(v) == 3 for v in cells.values()):
        tr_tab[c] = cells
if tr_tab:
    tcats = list(tr_tab)
    TRN["n_cat"] = len(tcats)
    TRN["means"] = {k: float(np.mean([tr_tab[c][k].mean() for c in tcats])) for k in ("train_t", "train_c", "fid_t", "fid_c")}
    TRN["train_c_vs_t"] = compare("train_t", "train_c", tr_tab, tcats)
    TRN["train_t_vs_held"] = compare("fid_t", "train_t", tr_tab, tcats)
    TRN["train_c_vs_held"] = compare("fid_c", "train_c", tr_tab, tcats)
    L = ["\\begin{tabular}{@{}l c c c c@{}}", "\\toprule",
         "Category & \\multicolumn{2}{c}{Held-out masks} & \\multicolumn{2}{c}{Training masks (CAS/NAS set)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & text & colour & text & colour \\\\", "\\midrule"]
    for c in tcats:
        L.append(f"{lab(c)} & " + " & ".join(f"{tr_tab[c][k].mean():.2f}" for k in ("fid_t", "fid_c", "train_t", "train_c")) + " \\\\")
    L += ["\\midrule", "\\textbf{Mean} & " + " & ".join(f"\\textbf{{{TRN['means'][k]:.3f}}}" for k in ("fid_t", "fid_c", "train_t", "train_c")) + " \\\\",
          "\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_cat_train_fidelity.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    N["train"] = TRN

# ---- control pools: post-hoc compositing and copy-paste baseline (review 2026-10-05)
CTRL = {}
ctl = {}
for c in CATS:
    if c not in det:
        continue
    cells = dict(det[c])
    cells.update({"comp_t": seeds("results_catcomp_cattext", c, "NAS"), "comp_c": seeds("results_catcomp_catcolor", c, "NAS"),
                  "paste": seeds("results_catpaste", c, "NAS")})
    # AnomalyDiffusion at 200 (results_ad) and 50 DDIM steps (results_cat_ad50), when available
    ad200, ad50 = seeds("results_ad", c, "NAS"), seeds("results_cat_ad50", c, "NAS")
    if ad200 is not None and ad50 is not None and len(ad200) == 3 and len(ad50) == 3:
        cells.update({"ad200": ad200, "ad50": ad50})
    if all(v is not None and len(v) == 3 for v in cells.values()):
        ctl[c] = cells
if ctl:
    kcats = list(ctl)
    CTRL["n_cat"] = len(kcats)
    CTRL["means"] = {k: float(np.mean([ctl[c][k].mean() for c in kcats])) for k in ("real", "nas_t", "nas_c", "comp_t", "comp_c", "paste")}
    for name, x, y in (("comp_t_vs_nas_t", "nas_t", "comp_t"), ("comp_c_vs_nas_c", "nas_c", "comp_c"), ("paste_vs_real", "real", "paste"),
                       ("nas_c_vs_paste", "paste", "nas_c"), ("nas_t_vs_paste", "paste", "nas_t"), ("comp_c_vs_paste", "paste", "comp_c")):
        CTRL[name] = compare(x, y, ctl, kcats)
    adc = [c for c in kcats if "ad50" in ctl[c]]
    if adc:
        CTRL["ad50_n_cat"] = len(adc)
        CTRL["means"]["ad200"] = float(np.mean([ctl[c]["ad200"].mean() for c in adc]))
        CTRL["means"]["ad50"] = float(np.mean([ctl[c]["ad50"].mean() for c in adc]))
        for name, x, y in (("ad50_vs_ad200", "ad200", "ad50"), ("ad50_vs_real", "real", "ad50"), ("nas_c_vs_ad50", "ad50", "nas_c")):
            CTRL[name] = compare(x, y, ctl, adc)
        f50 = {c: judge("ad50", "heldout", c) for c in jcats}
        f50 = {c: v for c, v in f50.items() if v is not None and len(v) == 3}
        if f50:
            CTRL["ad50_fid_mean"] = float(np.mean([v.mean() for v in f50.values()]))
            CTRL["ad50_fid_n"] = len(f50)
    L = ["\\begin{tabular}{@{}l c c c c c c@{}}", "\\toprule",
         "Category & Real only & \\multicolumn{2}{c}{Ali-AUG (NAS)} & \\multicolumn{2}{c}{Ali-AUG + compositing} & Copy-paste \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", " & & text & colour & text & colour & (NAS) \\\\", "\\midrule"]
    for c in kcats:
        vals = {k: ctl[c][k].mean() for k in ("real", "nas_t", "nas_c", "comp_t", "comp_c", "paste")}
        best = max(vals, key=vals.get)
        L.append(f"{lab(c)} & " + " & ".join((f"\\textbf{{{vals[k]:.3f}}}" if k == best else f"{vals[k]:.3f}") for k in vals) + " \\\\")
    L += ["\\midrule", "\\textbf{Mean} & " + " & ".join(f"\\textbf{{{CTRL['means'][k]:.3f}}}" for k in ("real", "nas_t", "nas_c", "comp_t", "comp_c", "paste")) + " \\\\",
          f"$p$ vs.\\ Ali-AUG, same carrier (categories / pairs) & -- & -- & -- & {fp(CTRL['comp_t_vs_nas_t']['p_cat'])} / {fp(CTRL['comp_t_vs_nas_t']['p_pairs'])} & {fp(CTRL['comp_c_vs_nas_c']['p_cat'])} / {fp(CTRL['comp_c_vs_nas_c']['p_pairs'])} & -- \\\\",
          f"$p$ vs.\\ real only (categories / pairs) & -- & {fp(N['nas_t_vs_real']['p_cat'])} / {fp(N['nas_t_vs_real']['p_pairs'])} & {fp(N['nas_c_vs_real']['p_cat'])} / {fp(N['nas_c_vs_real']['p_pairs'])} & -- & -- & {fp(CTRL['paste_vs_real']['p_cat'])} / {fp(CTRL['paste_vs_real']['p_pairs'])} \\\\",
          f"$p$ vs.\\ copy-paste (categories / pairs) & -- & {fp(CTRL['nas_t_vs_paste']['p_cat'])} / {fp(CTRL['nas_t_vs_paste']['p_pairs'])} & {fp(CTRL['nas_c_vs_paste']['p_cat'])} / {fp(CTRL['nas_c_vs_paste']['p_pairs'])} & -- & {fp(CTRL['comp_c_vs_paste']['p_cat'])} / {fp(CTRL['comp_c_vs_paste']['p_pairs'])} & -- \\\\",
          "\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "tab_controls.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    N["controls"] = CTRL

# ---- macros
mac = {}
if TRN:
    mac["TrainNcat"] = str(TRN["n_cat"])
    for k, v in TRN["means"].items():
        mac["TrainMean" + "".join(w.capitalize() for w in k.split("_"))] = f"{v:.3f}"
    for k in ("train_c_vs_t", "train_t_vs_held", "train_c_vs_held"):
        v = TRN[k]; tag = "Train" + "".join(w.capitalize() for w in k.split("_"))
        mac[tag + "Diff"], mac[tag + "CI"] = f"{v['diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
        mac[tag + "Ppairs"], mac[tag + "Pcat"], mac[tag + "Up"] = fpt(v["p_pairs"]), fpt(v["p_cat"]), f"{v['n_up']}/{v['n_cat']}"
if CTRL:
    mac["CtrlNcat"] = str(CTRL["n_cat"])
    for k, v in CTRL["means"].items():
        mac["CtrlMean" + "".join(w.capitalize() for w in k.split("_")).replace("200", "TwoHundred").replace("50", "Fifty")] = f"{v:.3f}"  # no digits in macro names
    if "ad50_fid_mean" in CTRL:
        mac["CtrlAdFiftyFid"], mac["CtrlAdFiftyFidN"] = f"{CTRL['ad50_fid_mean']:.3f}", str(CTRL["ad50_fid_n"])
    if "ad50_n_cat" in CTRL:
        mac["CtrlAdFiftyNcat"] = str(CTRL["ad50_n_cat"])
    for k, v in CTRL.items():
        if isinstance(v, dict) and "p_pairs" in v:
            tag = "Ctrl" + "".join(w.capitalize() for w in k.split("_")).replace("200", "TwoHundred").replace("50", "Fifty")
            mac[tag + "Diff"], mac[tag + "CI"] = f"{v['diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
            mac[tag + "Ppairs"], mac[tag + "Pcat"], mac[tag + "Up"] = fpt(v["p_pairs"]), fpt(v["p_cat"]), f"{v['n_up']}/{v['n_cat']}"
if RATIO:
    mac["RatioNcat"] = str(RATIO["n_cat"])
    for k, v in RATIO["means"].items():
        mac["RatioMean" + k.replace("05", "Half").replace("t", "T").replace("c", "C").replace("real", "Real")
            .replace("1", "One").replace("2", "Two").replace("5", "Five")] = f"{v:.3f}"
    for k, v in RATIO.items():
        if isinstance(v, dict) and "p_pairs" in v:
            tag = "Ratio" + "".join(w.replace("05", "Half").replace("1", "One").replace("2", "Two").replace("5", "Five")
                                    .capitalize() for w in k.split("_"))
            mac[tag + "Diff"], mac[tag + "Pcat"], mac[tag + "Up"] = f"{v['diff']:+.3f}", fpt(v["p_cat"]), f"{v['n_up']}/{v['n_cat']}"
            mac[tag + "CI"] = f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
if SAMN:
    for k in ("t", "c", "t_real", "c_real"):
        v = SAMN[k]
        tag = "Sam" + "".join(w.capitalize() for w in k.split("_"))
        mac[tag + "Diff"], mac[tag + "CI"] = f"{v['diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
        mac[tag + "Ppairs"], mac[tag + "Pcat"], mac[tag + "Up"] = fpt(v["p_pairs"]), fpt(v["p_cat"]), f"{v['n_up']}/{v['n_cat']}"
    for k, v in SAMN["means"].items():
        mac["SamMean" + "".join(w.capitalize() for w in k.split("_"))] = f"{v:.3f}" if not k.startswith(("iou", "acc")) else f"{v:.2f}"
    mac["SamNcat"] = str(SAMN["n_cat"])
    mac["SamFlagPct"] = f"{100 * (1 - (SAMN['means']['acc_t'] + SAMN['means']['acc_c']) / 2):.0f}"
    an = SAMN.get("analysis", {})
    for s, tag in (("text", "Text"), ("colour", "Colour"), ("ad", "Ad")):
        if an.get(f"{s}_iou") is not None:
            mac[f"SamHeldIou{tag}"] = f"{an[f'{s}_iou']:.2f}"
        v = an.get(f"{s}_t0.25")
        if v:
            mac[f"SamWrongRej{tag}"] = f"{100 * v['wrong_rejected']:.0f}"
            mac[f"SamRightRej{tag}"] = f"{100 * v['right_rejected']:.0f}"
            mac[f"SamAccHeld{tag}"] = f"{v['acc_rate']:.2f}"
if AD:
    for k, v in AD.items():
        if isinstance(v, dict) and "p_pairs" in v:
            tag = "Ad" + "".join(w.capitalize() for w in k.split("_"))
            mac[tag + "Diff"], mac[tag + "CI"] = f"{v['diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
            mac[tag + "Ppairs"], mac[tag + "Pcat"], mac[tag + "Up"] = fpt(v["p_pairs"]), fpt(v["p_cat"]), f"{v['n_up']}/{v['n_cat']}"
    for k, v in AD["means_full"].items():
        mac["AdFull" + k.capitalize()] = f"{v:.3f}"
    for k, v in AD["means_clean"].items():
        mac["AdClean" + k.capitalize()] = f"{v:.3f}"
    for k, v in AD["fid"].items():
        mac["AdFid" + k.replace("fid_", "").capitalize()] = f"{v:.3f}"
    for k, v in AD["cas"].items():
        mac["AdCas" + k.replace("cas_", "").replace("ad_", "").capitalize()] = f"{v:.3f}"
    mac["AdSeen"], mac["AdTestTotal"] = str(AD["seen"]), str(AD["test_total"])
    mac["AdSeenPct"] = f"{100 * AD['seen'] / AD['test_total']:.0f}"
    mac["AdUnseen"] = str(AD["test_total"] - AD["seen"])
    mac["AdNcat"], mac["AdNclean"], mac["AdNjcat"] = str(AD["n_cat"]), str(AD["n_clean"]), str(AD["n_jcat"])
    mac["AdNobj"], mac["AdNtex"] = str(AD["n_obj"]), str(AD["n_tex"])
    for pk, tag in (("AnomalyDiffusion", "Ad"), ("Ali-AUG text", "T"), ("Ali-AUG colour", "C")):
        if "ic_lpips" in AD:
            mac["IcLpips" + tag] = f"{AD['ic_lpips'][pk]:.3f}"
        if "comp_lpips" in AD and pk != "AnomalyDiffusion":
            mac["CompLpips" + tag + "Obj"], mac["CompLpips" + tag + "Tex"] = f"{AD['comp_lpips']['objects'][pk]:.3f}", f"{AD['comp_lpips']['textures'][pk]:.3f}"
    for pk, tag in (("AnomalyDiffusion", "Ad"), ("Ali-AUG text", "Text"), ("Ali-AUG colour", "Colour")):
        mac[f"Lpips{tag}Obj"] = f"{AD['lpips']['objects'][pk]:.3f}"
        mac[f"Lpips{tag}Tex"] = f"{AD['lpips']['textures'][pk]:.3f}"
        mac[f"Sharp{tag}Obj"] = f"{AD['sharp'][pk]['objects']:.2f}"
        mac[f"Sharp{tag}Tex"] = f"{AD['sharp'][pk]['textures']:.2f}"
for k in ("nas_t_vs_real", "nas_c_vs_real", "cas_t_vs_real", "cas_c_vs_real", "nas_c_vs_t", "fid_c_vs_t", "swap_c_vs_t"):
    v = N[k]
    tag = "".join(w.capitalize() for w in k.replace("_vs_", " vs ").replace("_", " ").split())
    mac[tag + "X"] = f"{v['x']:.3f}"
    mac[tag + "Y"] = f"{v['y']:.3f}"
    mac[tag + "Diff"] = f"{v['diff']:+.3f}"
    mac[tag + "CI"] = f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
    mac[tag + "Ppairs"] = fpt(v["p_pairs"])
    mac[tag + "Pcat"] = fpt(v["p_cat"])
    mac[tag + "Up"] = f"{v['n_up']}/{v['n_cat']}"
    mac[tag + "Npairs"] = str(v["n_pairs"])
mac["CatN"], mac["CatJN"], mac["CatCeil"] = str(N["n_cat"]), str(N["n_jcat"]), f"{N['ceil_mean']:.2f}"
mac["CatChance"] = f"{N['chance_mean']:.2f}"
# single adapter for all categories (split 0), same judges, restricted to the categories of the judge table
for var, tag in (("alltext", "Text"), ("allcolor", "Colour")):
    for mode, mt in (("heldout", "Fid"), ("swap", "Swap")):
        fs = sorted(glob.glob(os.path.join(a.live, "results_judge", f"{var}_{mode}_split0_seed*.json")))
        if not fs:
            continue
        per = {}
        for f in fs:
            for c, x in json.load(open(f))["per_category"].items():
                per.setdefault(c, []).append(x["fidelity"] if isinstance(x, dict) else x)
        v = np.mean([np.mean(per[c]) for c in jcats if c in per])
        mac[f"Single{mt}{tag}"] = f"{v:.3f}"
        N[f"single_{mode}_{var}"] = float(v)
if f"aliaug_{TGPU}" in tim and f"ad_{TGPU}" in tim:
    to, ta = tim[f"aliaug_{TGPU}"], tim[f"ad_{TGPU}"]
    mac["TimeOurs"] = f"{to['s_per_img_mean']:.2f}"
    mac["TimeAD"] = f"{ta['s_per_img_mean']:.2f}"
    mac["TimeRatio"] = f"{ta['s_per_img_mean'] / to['s_per_img_mean']:.1f}"
    mac["TimeGPU"] = to["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    if mac_ad50 is not None:
        mac["TimeADFifty"] = f"{mac_ad50:.2f}"
        mac["TimeRatioFifty"] = f"{mac_ad50 / to['s_per_img_mean']:.1f}"
with open(os.path.join(a.out, "cat_numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by prep/t23/tables_cat.py -- do not edit\n")
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump({"numbers": N, "macros": mac, "categories": cats, "judge_categories": jcats},
          open(os.path.join(a.out, "cat_numbers.json"), "w"), indent=1)
print("categories", len(cats), "judge", len(jcats))
for k in ("nas_t_vs_real", "nas_c_vs_real", "nas_c_vs_t", "fid_c_vs_t", "swap_c_vs_t"):
    v = N[k]
    print(f"{k}: {v['x']:.3f}->{v['y']:.3f} {v['diff']:+.3f} CI[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] p {v['p_pairs']:.4f}/{v['p_cat']:.4f} up {v['n_up']}/{v['n_cat']}")
