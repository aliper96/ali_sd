# Experiments of the paper (MVTec-AD studies)

Everything in this folder was used, as is, to produce the numbers of the paper. Launch scripts (`*.sh`)
are the exact commands run on the authors' machines and contain their paths; the Python entry points
take explicit arguments. No MVTec-AD image is redistributed: `manifests/` lists, for every split, the
file names of the real defects used (MVTec-AD naming, `<category>_<type>_<index>.png`) and the prompt
attached to each; images are rebuilt from the official MVTec-AD release with the split builders.

| Paper item | Produced by | Inputs |
|---|---|---|
| Tile splits (5 × 60/24) | `t23/build_aliaug_splits.py` | MVTec-AD *tile* |
| Per-category splits (14 categories, seed 0) | `t23/build_mvtec_all_splits.py` then `t23/build_cat_splits.py` | MVTec-AD |
| Colour-coded masks (same splits) | `t23/build_color_splits.py` (`--white` = no label carrier) | a split |
| Generator training (10,000 steps, seed = split) | `t23/train_gen.py --fix-sched` (wraps `train_ali2ali.py`) | a split |
| Synthetic pool (one image per training mask, fixed per-image seed) | `t23/gen_syn.py` | a checkpoint |
| Ratio sweep pools (0.5/2/5×) | `t23/gen_syn.py` + `t23/run_ratio.sh`, `t23/cola_ratio2.sh` | |
| YOLOv8-seg datasets (D_S, D_S_AUG, CAS, NAS) | `t23/build_yolo.py` | real split + pool |
| Detector runs, last-epoch evaluation | `eval/run_downstream_yolo.py`, epochs/imgsz per study in `eval/epochs_override.json` | |
| Label-fidelity judge (EfficientNet-B0), ceiling, swap test | `t23/judge.py` | real crops + held-out masks |
| SAM step (box prompt, IoU ≥ 0.25) | `t23/sam_box.py`, `t23/sam_analysis.py`, `t23/run_samstep*.sh` | SAM 3 official checkpoint |
| AnomalyDiffusion under the same protocol | `t23/ad/` (`gen_ad.py` runs inside the official AD repo with `patch_ad.py`) | AD released checkpoint |
| Preservation outside the mask (LPIPS, sharpness) | `t23/ad/preserve_lpips.py`, `t23/ad/preserve_metrics.py` | |
| Inference cost | `t23/time_infer.py` (Ali-AUG), timing block of `t23/ad/gen_ad.py` (AD) | |
| Tile tables | `t23/tables_tile.py` | `results/results/`, `results/results_judge/` |
| All per-category tables and the numeric macros of the text | `t23/tables_cat.py --live results --splits <cat splits> --out tables` | `results/` |
| Dataset statistics table | `prep/build_dataset_stats.py` | MVTec-AD |
| Failure figure | `t23/fig_failures.py` / `prep/build_failure_fig.py` | judge outputs |

`t23/PLAN.md` is the dated experiment plan (protocol, amendments and author notes) written before the
runs; `results/` holds every result file (detector CSVs, judge JSONs, SAM, timing, preservation) that
`tables_*.py` read, so every table of the paper can be regenerated without re-training:

```
python experiments/t23/tables_cat.py --live experiments/results --splits <per-category splits> --out out/
python experiments/t23/tables_tile.py --help
```

Seeds: generator seed = split index (tile) or 0 (categories); per-image generation seed = CRC32 of the
file name; detector seeds 0, 1, 2; judge seeds 0, 1, 2; bootstrap 10,000 resamples. See `PLAN.md`.
