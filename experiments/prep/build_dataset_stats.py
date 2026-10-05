"""Count MVTec-AD dataset stats for tab:dataset_stats (Phase 0, no GPU)."""
import os, glob, collections

ROOT = r"C:\Users\aliha\Desktop\PHD\FIRSTPAPER-FINAL"
MVTEC = os.path.join(ROOT, "mvtec_anomaly_detection")
LORA_IN = os.path.join(ROOT, "dataset", "mvtec_defects", "mvtec", "inputs")

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

def count_imgs(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith(IMG_EXT))

cats = sorted([d for d in os.listdir(MVTEC)
               if os.path.isdir(os.path.join(MVTEC, d))])

# LoRA defective samples per category (id = "<cat>__<defect>__<stem>")
lora_by_cat = collections.Counter()
for d in os.listdir(LORA_IN):
    if os.path.isdir(os.path.join(LORA_IN, d)):
        cat = d.split("__")[0]
        lora_by_cat[cat] += 1

rows = []
tot = collections.Counter()
for cat in cats:
    ctrain_good = count_imgs(os.path.join(MVTEC, cat, "train", "good"))
    ctest_good  = count_imgs(os.path.join(MVTEC, cat, "test", "good"))
    test_dir = os.path.join(MVTEC, cat, "test")
    defects = sorted([x for x in os.listdir(test_dir)
                      if os.path.isdir(os.path.join(test_dir, x)) and x != "good"]) \
              if os.path.isdir(test_dir) else []
    ctest_anom = sum(count_imgs(os.path.join(test_dir, x)) for x in defects)
    gt_dir = os.path.join(MVTEC, cat, "ground_truth")
    cmasks = sum(count_imgs(os.path.join(gt_dir, x)) for x in defects) \
             if os.path.isdir(gt_dir) else 0
    clora = lora_by_cat.get(cat, 0)
    rows.append((cat, len(defects), ctrain_good, ctest_good, ctest_anom, cmasks, clora))
    tot["defects"] += len(defects); tot["train_good"] += ctrain_good
    tot["test_good"] += ctest_good; tot["test_anom"] += ctest_anom
    tot["masks"] += cmasks; tot["lora"] += clora

# ---- readable table ----
hdr = f"{'category':<12}{'#deftypes':>10}{'train/good':>12}{'test/good':>11}{'test/anom':>11}{'GT masks':>10}{'LoRA smpl':>11}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r[0]:<12}{r[1]:>10}{r[2]:>12}{r[3]:>11}{r[4]:>11}{r[5]:>10}{r[6]:>11}")
print("-"*len(hdr))
print(f"{'TOTAL':<12}{tot['defects']:>10}{tot['train_good']:>12}{tot['test_good']:>11}{tot['test_anom']:>11}{tot['masks']:>10}{tot['lora']:>11}")

# ---- LaTeX ----
print("\n\n% ===== LaTeX: tab:dataset_stats =====")
latex = r"""\begin{table*}[ht]
\centering
\caption{MVTec-AD dataset statistics per category. \textbf{Normal (train)} = real
defect-free images (used as unpaired clean references and for the real-data
detector baseline); \textbf{Normal (test)} and \textbf{Anomalous (test)} = the
untouched real test fold (evaluation only, never used for generation);
\textbf{GT masks} = ground-truth annotation masks; \textbf{LoRA samples} = real
anomalous images (+mask+prompt) used to train the generator. Synthetic counts
per split are reported in Table~\ref{tab:synth_construction} (Phase~1).}
\label{tab:dataset_stats}
\renewcommand{\arraystretch}{1.05}
\setlength{\tabcolsep}{5pt}
\begin{small}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Category} & \textbf{\#Defect types} & \textbf{Normal (train)} & \textbf{Normal (test)} & \textbf{Anomalous (test)} & \textbf{GT masks} & \textbf{LoRA samples} \\
\midrule
"""
for r in rows:
    cat = r[0].replace("_", r"\_")
    latex += f"{cat} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} \\\\\n"
latex += r"\midrule" + "\n"
latex += (f"\\textbf{{Total}} & {tot['defects']} & {tot['train_good']} & "
          f"{tot['test_good']} & {tot['test_anom']} & {tot['masks']} & {tot['lora']} \\\\\n")
latex += r"""\bottomrule
\end{tabular}
\end{small}
\end{table*}
"""
print(latex)
