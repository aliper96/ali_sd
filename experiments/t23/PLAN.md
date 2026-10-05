# Re-run of Tables 2-3 (Tile, CAS/NAS) — protocol, fixed before seeing results

Why: the published Tables 2-3 used a Tile set with test == train (md5-verified 2026-08-24).

## Data
- `D:/DATASET/aliaug_splits/split_{0..4}` (built by `D:/DATASET/build_aliaug_splits.py`):
  K = 5 random splits of the 84 MVTec-AD Tile defect masks, stratified by defect type and by source
  image; 60 train / 24 test each; 5 types (crack, glue_strip, gray_stroke, oil, rough). The published
  version had 69 masks and no `rough` (type silently skipped by the old builder).
- Clean inputs (train_C) come from MVTec `train/good` only.

## Generator (one per split, trained on the TRAIN fold only — L1)
- Released code `ali_sd_public/train_ali2ali.py`, run through `prep/t23/train_gen.py`
  (wandb neutralised, seeds fixed; method unchanged).
- Recipe decided 2026-09-26: **no conditioning dropout** (as in the published results; the paper's
  0.3 was never in the committed code), **10,000 steps**, final checkpoint (no selection),
  lambda_lpips 10 / l2 10 / gan 2.5 / clipsim 5, LoRA 8/4, lr 5e-4, batch 1, 512 px. Seed = split index.
- The published Tile checkpoint had 84,501 steps; 10k was chosen for compute and is declared.

## Synthetic pool (S5, L2, L3)
- `gen_syn.py`: one image per TRAIN triplet (mask, clean, prompt) → N_syn = N_real_train = 60.
  Fixed per-sample seed (crc32 of the file name). Class = the prompt's defect (weak label).

## Downstream (S1-S4)
- `build_yolo.py` → binary (1 class) and multiclass (5 classes) YOLO-seg sets.
- `run_downstream_yolo.py`: D_S (no aug), D_S_AUG (ultralytics default aug), CAS (synthetic only),
  NAS (real + synthetic 1:1; CAS/NAS also use default aug). yolov8n-seg, 100 epochs, imgsz 640,
  seeds 0,1,2. Evaluated on the held-out real test fold with **last.pt** (best.pt would be selected on
  the test fold).
- 5 splits × 3 seeds = 15 runs per cell. `aggregate.py`: mean ± std, bootstrap 95 % CI, paired
  Wilcoxon vs D_S_AUG (pairs = split × seed).
- Table 2 = binary, box metrics (B). Table 3 = multiclass, mask metrics (M).

## Where it runs
A100 VM, `/home/aliha/t23`: `train_gen_split_k.log`, `logs/run_split_k.log`,
results in `results/split_k/<task>/<P>.csv`. Splits 0-2 train in parallel; `cola_a100.sh` starts 3 and 4
when 0 and 1 finish; `run_split.sh k` chains generation → YOLO for each split.

## Full-MVTec-AD study (alltext / allcolor / allreal) — decisions (2026-09-28/29)
- Splits: `prep/t23/build_mvtec_all_splits.py`, K = 3, 880 train / 378 test defects per split (73 types,
  15 categories), 512 px, hybrid clean input (textures: random good image; objects: the defective image
  with the masked region inpainted, cv2 Telea on the 7-px-dilated mask).
- Generator: ONE adapter for all categories per split and carrier, 20,000 steps, otherwise as tile.
- Colour: palette of 8 colours indexed by type within its category (`mvtec_names.py`); tile keeps its colours.
- Judge: one EfficientNet-B0 per category (types within the category); toothbrush (1 type) excluded.
- **Detector epochs: 30 (not 100) for every condition of this study** — real-only D_S/D_S_AUG
  (`run_allreal.sh`), text and colour CAS/NAS. Decided 2026-09-29 12:30 before any full-MVTec detector
  result existed: at 100 epochs each run took 100-200 min on the shared A100 (>180 h in total).
  Implemented in `eval/epochs_override.json`; the epochs used are written to every CSV row. The first
  real-only run, started at 100 epochs, was discarded (`logs/allreal_0_100ep_descartado.log`).
- Tile ablation (abl_noskip / abl_noclip / abl_nogan / abl_nolabel): colour carrier, K = 3, 10k steps,
  100 detector epochs (same as tile).
- **Detector scope of the full-MVTec study (2026-09-29 15:10, still before any of its detector results):**
  the GPU is shared with three generator trainings and YOLO ran at ~5 min/epoch (not CPU-bound: load 5.5/12).
  To keep the study within ~30 GPU-hours: multiclass task only (73 classes — where the label carrier
  matters; the binary task is covered by tile), protocols D_S_AUG / CAS / NAS only (no D_S), image size
  512 = the native size of these splits (640 only upsampled), 30 epochs, 3 detector seeds. Rules in
  `eval/epochs_override.json` (skipped runs write no CSV; epochs and imgsz are recorded in every row).
  The real-only runs started under the previous rules were discarded (`logs/allreal_*_descartado2.log`).

## Per-category LoRA study (decided by the author 2026-09-29 21:30, after seeing single-adapter samples)
- The single adapter for all 15 categories produced weak defects on objects (visual check of split 0).
  Its splits 1-2 are cancelled; split 0 is kept as a data point. The real-only full-MVTec reference is
  kept for split 0 only.
- One LoRA per category and carrier (14 categories; tile already done with K = 5): `run_cat.sh`, split 0
  of the full-MVTec splits cut per category (`build_cat_splits.py`: same samples and inpainted clean
  inputs), same recipe as tile (10k steps, seed 0). Statistics pair 14 categories x 3 detector seeds.
- Detector: multiclass within the category, D_S_AUG (once per category) / CAS / NAS, 50 epochs, 512 px
  (`epochs_override.json` rule "/results_cat"; smaller data than tile, decided before any result).
- Judge: held-out + swap per category (toothbrush has one type: no judge).
- Queue `cola_cat.sh`: per-category jobs first, then the remaining tile ablations; SAM after it.

## Synthetic:real ratio sweep (review theme I) — fixed 2026-10-02 19:00, before any ratio result
- Per-category LoRAs of the study above, final checkpoints, both carriers, all 14 categories; no retraining.
- Pool: `gen_syn --per-mask 5` (textures: 4 extra clean images of the same train fold; objects: the sample's own
  inpainted clean image with 4 other noise seeds — tests the one-step diversity limit).
- NAS at ratios 0.5 / 2 / 5 (ratio 1 = the per-category result above), 3 detector seeds, same detector settings
  (50 ep, 512 px). Every ratio is reported; no ratio is selected on the test fold.
- Statistics: per ratio, paired Wilcoxon vs D_S_AUG (category x seed) and at category level, as above.
- Runs one chain at a time alongside the main queue: `cola_ratio.sh` -> `run_ratio.sh`, results in
  `results_catratio_<carrier>/<cat>/multiclass/NAS.csv`, end marker `logs/ratio_all.done`.

## AnomalyDiffusion baseline under our protocol + efficiency table — fixed 2026-10-02 22:00, before any result
- Why re-run: their released checkpoints were trained on THEIR split (lowest-ID third), which overlaps our test
  fold, so they cannot be used. Their official evaluation also has issues we do not inherit (verified in their
  code 2026-10-02): train-localization.py and train-classification.py evaluate on the test 2/3 every epoch and
  keep the checkpoint with the best TEST metric; test-classification accuracy is computed on the last batch of
  100 only; the sample at index len//3 of each type is in both their train and test sets; the released
  synthetic data were filtered manually ("filtered out some data with poor generation effects", README).
  Their published IS / IC-LPIPS were not reproducible from the released data (our earlier check; their issues
  #100, #119).
- Training: official code (`ad/run_ad.sh`), recipe of their paper (300k iterations, batch 4, lr 5e-3, one model
  for all types), on our split-0 train fold of the 14 categories only (`ad/build_ad_data.py`, `ad/patch_ad.py`
  removes their 1/3 split; nothing else changed). Final checkpoint, no selection.
- Generation (`ad/gen_ad.py`): one image per train mask with OUR clean inputs and masks (same as Ali-AUG), their
  defaults (DDIM 200, eta 1, 256 px, upsampled to 512), adaptive attention for texture categories. No filtering.
- Downstream: identical to the per-category study (multiclass, NAS/CAS, 50 ep, 512 px, 3 detector seeds,
  last.pt). Judge (held-out masks) as for Ali-AUG.
- Efficiency table: s/image (mean ± std over 100 calls after 10 warm-up, batch 1), images/s, peak GPU memory,
  parameters, steps — Ali-AUG (`time_infer.py`) and AnomalyDiffusion (`gen_ad.py --time`), on the IDLE A100
  only (scripts refuse if the GPU is busy). Other methods: published step counts/times, marked as such.

## AnomalyDiffusion: protocol change (decided by the author 2026-10-03 ~09:00, before any AD result)
- Re-training AD (300k it, 2-3 days) is cancelled; the RELEASED checkpoints are used instead (official
  Google Drive of the repo: logs/anomaly-checkpoints/{embeddings.pt, spatial_encoder.pt}), run on the local
  RTX 5090. Generation is unchanged (`ad/gen_ad.py`: our train masks and clean inputs, DDIM 200, 256 -> 512 px,
  no filtering); downstream and judge identical to Ali-AUG (`ad/run_ad_cat.sh` on the A100; it skips
  generation when the uploaded pools exist).
- Consequence, declared in the paper: their generator was trained on the lowest-ID third of every defect type,
  which contains 130 of our 354 split-0 test images (37 %; per category in t23_runs/live/ad_overlap_split0.json)
  and 304 of our 820 train images. This leaks test defects into the AD synthetic pool and FAVOURS AD; the
  comparison is therefore conservative for Ali-AUG (if Ali-AUG >= AD, the conclusion holds; if AD > Ali-AUG it
  is inconclusive). The held-out judge for AD is equally affected and is reported with the same caveat.
- Loading: torch.load with weights_only kept on; only torch.nn.ParameterDict / Parameter are allow-listed.

## Secondary analysis: test images unseen by AnomalyDiffusion — fixed 2026-10-03 13:00, before any result on it
- Motivation: AD's released generator saw 130/354 of our test images. To separate generation quality from that
  leak, the four detectors (real-only D_S_AUG, Ali-AUG text NAS, Ali-AUG colour NAS, AD NAS) are re-trained
  with the same settings and evaluated ONLY on the 224 test images AD never saw (`ad/run_clean.sh`,
  results_clean/<cond>/<cat>/multiclass). Same statistics (pairs category x seed; category level). Both the
  full-test and the unseen-test results are reported.

## SAM 3 study (author's priority, 2026-10-04 09:30) — fixed before any SAM result on MVTec; run locally on the 5090
Question: does SAM 3, as the automatic step of generate -> SAM -> human check, help? Data: held-out (test-fold)
synthetic pools of split 0 — Ali-AUG text, Ali-AUG colour, AnomalyDiffusion — and the REAL test defects as reference,
14 categories (judge-based analyses: 13, no toothbrush). Script `sam_mvtec.py`, analysis `sam_analysis.py`.
- Prompts per category: every defect-type name of the category (underscores -> spaces) plus "defect"; instances with
  score >= 0.3 kept (as in sam_label.py). Predicted defect region = union of kept instances.
- WHERE: IoU(predicted region, requested mask); hit = predicted region covers >= 10 % of the requested mask;
  bleed = fraction of the predicted region outside the requested mask dilated by 15 px.
- WHAT: SAM label = type whose best instance overlapping the requested mask has the highest score.
- Primary "does it help" analysis (no new training): SAM as verifier, accept an image iff hit. Report acceptance rate,
  judge label fidelity of accepted vs all images (paired over categories), the fraction of judge-wrong labels that SAM
  rejects, and the fraction of judge-right labels it rejects. Secondary: agreement of the SAM label with the judge vs
  agreement of the requested label with the judge. Thresholds 10 % / 0.3 are fixed here; other values only as sensitivity.
- AMENDMENT 2026-10-04 10:40 (before any aggregate SAM analysis; seen: two debug images and the per-category
  one-line summaries of bottle (4 sources) and cable (2 sources), all ~0 hits): SAM 3 does not respond to the MVTec
  type names ("broken large", "contamination") nor to "defect", but does to common nouns ("hole" 0.90 on a real
  bottle defect, "crack" 0.70 on a real hazelnut crack). The prompt set is therefore extended with a fixed generic
  vocabulary used for every category: hole, crack, scratch, stain, dent, spot, contamination, broken, missing part,
  cut, tear, bent, hair, thread, fold. WHERE uses the union over type names + vocabulary + "defect"; the SAM label
  (WHAT) still uses the type names only. The discarded first run is not reported.
- AMENDMENT 2, author's direction 2026-10-04 11:00 ("help SAM: we already have the mask; then box/point to polygon"),
  before any aggregate result of it (seen: 12 debug real images, IoU 0.01-0.84). REPLACES the text-prompt design:
  * SAM-box step: prompt SAM 3 with the bounding box of the REQUESTED mask (padded 10 %, min 16 px, positive box);
    SAM mask = returned instance with the highest IoU with the requested mask; its polygons become the label.
  * Verification: an image is ACCEPTED iff IoU(SAM mask, requested mask) >= 0.25 (sensitivity: 0.1 and 0.5).
  * (1) reference on the REAL test defects: IoU(SAM mask, ground-truth mask); (2) held-out synthetic pools (text,
    colour, AD): IoU with the requested mask, acceptance rate, judge fidelity of accepted vs all images, share of
    judge-wrong images rejected and of judge-right images rejected (13 categories with a judge);
  * (3) downstream "does the SAM step help": NAS where the synthetic set is passed through the SAM step (rejected
    images dropped, accepted ones labelled with the SAM polygon), vs. the current NAS (requested masks), per category
    and carrier, 3 detector seeds, same detector settings (50 ep, 512 px), results_samstep_<carrier>/<cat>; same
    statistics as the per-category study. Scripts: sam_box.py (local 5090), build_yolo.py --syn-masks/--drop-missing.
- Author 2026-10-04: SAM is used for the CONTOUR only (box of the requested mask -> SAM mask -> polygon); the type stays
  the requested one. The verification analysis is reported as a secondary observation (SAM-box does not check type).
  Ablations of splits 1-2 postponed; split-0 ablations kept.
