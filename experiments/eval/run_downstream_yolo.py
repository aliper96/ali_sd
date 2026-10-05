"""
run_downstream_yolo.py — Downstream defect-detection/segmentation evaluation.

Implements the CAS / NAS protocols (Ravuri & Vinyals) plus the two real-only
baselines, trains YOLOv8 (detection) or YOLOv8-seg (segmentation), and reports
metrics over MULTIPLE SEEDS with mean +/- std and a bootstrap 95% CI.

This directly addresses the reviewers:
  - F  : repeated runs + std + CIs (R3 Q3, "needs statistician").
  - E  : the small-split sensitivity (run with several --seeds / --val-yaml folds).
  - I  : the synthetic-to-real ratio sweep (pass several --ratio values).
The generator that produced the synthetics is
irrelevant to this script — it only consumes generated images + labels.

Protocols:
  D_S      : real train split only, NO augmentation (ultralytics aug disabled).
  D_S_AUG  : real train split + conventional augmentation (ultralytics defaults).
  CAS      : synthetic images ONLY (Classification Accuracy Score).
  NAS      : synthetic mixed with real train images at --ratio (Naive Aug. Score).
ALL protocols are validated on the SAME held-out REAL split (--val-yaml / val).
Synthetic images are NEVER added to validation — prevents the leakage the paper
must rule out.

Expected inputs (YOLO format):
  --real-images-dir / --real-labels-dir   real TRAIN images + labels (.txt)
  --val-images-dir  / --val-labels-dir     held-out REAL val/test (never augmented)
  --syn-images-dir  / --syn-labels-dir     synthetic images + labels (prompt-assigned
                                           class; segment task -> polygon labels)
  --names          class names in order (e.g. crack glue_strip gray_stroke oil)

Usage (NAS ratio sweep, 5 seeds, segmentation):
    python eval/run_downstream_yolo.py \\
        --task segment --model yolov8n-seg.pt \\
        --real-images-dir data/real/train/images --real-labels-dir data/real/train/labels \\
        --val-images-dir  data/real/val/images   --val-labels-dir  data/real/val/labels \\
        --syn-images-dir  data/syn/images         --syn-labels-dir  data/syn/labels \\
        --names crack glue_strip gray_stroke oil \\
        --protocol NAS --ratio 0.5 1 2 5 --seeds 0 1 2 3 4 \\
        --epochs 100 --output-csv results/downstream_nas.csv

Dependencies:
    pip install ultralytics pandas numpy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from stats_utils import aggregate_runs  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _imgs(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())  # absolute: a relative target would dangle
    except OSError:
        shutil.copy2(src, dst)  # Windows without symlink privilege


def _label_for(img: Path, labels_dir: Path) -> Optional[Path]:
    cand = labels_dir / (img.stem + ".txt")
    return cand if cand.exists() else None


def build_train_set(
    work: Path,
    protocol: str,
    real_images: Path,
    real_labels: Path,
    syn_images: Path,
    syn_labels: Path,
    ratio: float,
    seed: int,
):
    """
    Materialise a YOLO train folder (images/ + labels/) for the given protocol.
    Returns the number of (real, synthetic) images placed.
    """
    import numpy as np

    timg = work / "train" / "images"
    tlbl = work / "train" / "labels"
    for d in (timg, tlbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_real = n_syn = 0

    # Real images (all protocols except CAS)
    if protocol in ("D_S", "D_S_AUG", "NAS"):
        for img in _imgs(real_images):
            lbl = _label_for(img, real_labels)
            if lbl is None:
                continue
            _link_or_copy(img, timg / img.name)
            _link_or_copy(lbl, tlbl / lbl.name)
            n_real += 1

    # Synthetic images (CAS = all synthetics; NAS = ratio * n_real)
    if protocol in ("CAS", "NAS"):
        syn = _imgs(syn_images)
        syn = [s for s in syn if _label_for(s, syn_labels) is not None]
        rng.shuffle(syn)
        if protocol == "NAS":
            k = int(round(ratio * max(n_real, 1)))
            syn = syn[:k]
        for img in syn:
            lbl = _label_for(img, syn_labels)
            # prefix to avoid name clashes with real images
            _link_or_copy(img, timg / f"syn_{img.name}")
            _link_or_copy(lbl, tlbl / f"syn_{lbl.stem}.txt")
            n_syn += 1

    return n_real, n_syn


def write_data_yaml(work: Path, val_images: Path, names: List[str]) -> Path:
    yaml_path = work / "data.yaml"
    lines = [
        f"path: {work.resolve()}",
        "train: train/images",
        f"val: {val_images.resolve()}",
        f"nc: {len(names)}",
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def run_one(
    task: str,
    model_name: str,
    data_yaml: Path,
    protocol: str,
    seed: int,
    epochs: int,
    imgsz: int,
    project: Path,
) -> dict:
    """Train + validate one run; return a metric dict."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    # D_S = no augmentation; others use ultralytics defaults (conventional aug).
    aug_off = dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
        mosaic=0.0, mixup=0.0, erasing=0.0,
    )
    train_kwargs = dict(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, seed=seed,
        project=str(project), name=f"{protocol}_seed{seed}", exist_ok=True,
        verbose=False,
    )
    if protocol == "D_S":
        train_kwargs.update(aug_off)

    model.train(**train_kwargs)
    # Evaluate the LAST epoch, not best.pt: ultralytics picks best.pt by fitness on the
    # val split, which here IS the held-out test set -- selecting on it would leak.
    last = project / f"{protocol}_seed{seed}" / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(last)
    model = YOLO(str(last))
    metrics = model.val(data=str(data_yaml), split="val", verbose=False)

    row = {"protocol": protocol, "seed": seed}
    box = getattr(metrics, "box", None)
    if box is not None:
        row.update({
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        })
    if task == "segment":
        seg = getattr(metrics, "seg", None)
        if seg is not None:
            row.update({
                "seg_mAP50": float(seg.map50),
                "seg_mAP50_95": float(seg.map),
                "seg_precision": float(seg.mp),
                "seg_recall": float(seg.mr),
            })
    return row


def _cli():
    p = argparse.ArgumentParser(description="Downstream YOLO eval (CAS/NAS/D_S/D_S_AUG) with seeds + CIs.")
    p.add_argument("--task", choices=["detect", "segment"], default="segment")
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--real-images-dir", required=True)
    p.add_argument("--real-labels-dir", required=True)
    p.add_argument("--val-images-dir", required=True)
    p.add_argument("--val-labels-dir", required=True,
                   help="(used only to document the val labels location; YOLO reads them "
                        "alongside val images via the standard images/->labels/ mapping)")
    p.add_argument("--syn-images-dir", default=None)
    p.add_argument("--syn-labels-dir", default=None)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--protocol", choices=["D_S", "D_S_AUG", "CAS", "NAS"], required=True)
    p.add_argument("--ratio", type=float, nargs="+", default=[1.0],
                   help="NAS synthetic:real ratio(s). Pass several for the ratio sweep (review I).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workdir", default="results/_yolo_work")
    p.add_argument("--output-csv", default="results/downstream.csv")
    args = p.parse_args()

    if args.protocol in ("CAS", "NAS") and not (args.syn_images_dir and args.syn_labels_dir):
        p.error("CAS/NAS require --syn-images-dir and --syn-labels-dir")

    import pandas as pd

    # Epoch override for the full-MVTec-AD study (decided 2026-09-29, before any of its detector
    # results existed): 880 real / 1760 NAS images x 100 epochs made each run take 100-200 min on the
    # shared A100. For every condition of that study (real-only, text, colour) the detector is trained
    # for 30 epochs instead; tile keeps 100. The epochs actually used are written to each CSV row.
    # Rules (2026-09-29 15:10, also before any full-MVTec result): for that study only the multiclass
    # task and the D_S_AUG / CAS / NAS protocols are run, at the native 512 px of its splits; runs
    # matching a "skip" rule exit without writing a CSV. A rule applies when ALL its "match" substrings
    # occur in the output-CSV path.
    ov = Path(__file__).resolve().parent / "epochs_override.json"
    if ov.exists():
        import json
        path = str(args.output_csv).replace("\\", "/")
        for rule in json.load(open(ov))["rules"]:
            if all(m in path for m in rule["match"]):
                if rule.get("skip"):
                    print(f"[override] skipped by rule {rule['match']} ({ov.name})")
                    return
                for k in ("epochs", "imgsz"):
                    if k in rule:
                        print(f"[override] {k} {getattr(args, k)} -> {rule[k]} (rule {rule['match']})")
                        setattr(args, k, int(rule[k]))

    work_root = Path(args.workdir)
    project = Path(args.output_csv).parent / "_yolo_runs"
    ratios = args.ratio if args.protocol == "NAS" else [0.0]

    all_rows = []
    metric_keys = ["mAP50", "mAP50_95", "precision", "recall"]
    if args.task == "segment":
        metric_keys += ["seg_mAP50", "seg_mAP50_95", "seg_precision", "seg_recall"]

    for ratio in ratios:
        for seed in args.seeds:
            work = work_root / f"{args.protocol}_r{ratio}_s{seed}"
            n_real, n_syn = build_train_set(
                work=work, protocol=args.protocol,
                real_images=Path(args.real_images_dir), real_labels=Path(args.real_labels_dir),
                syn_images=Path(args.syn_images_dir) if args.syn_images_dir else Path("."),
                syn_labels=Path(args.syn_labels_dir) if args.syn_labels_dir else Path("."),
                ratio=ratio, seed=seed,
            )
            data_yaml = write_data_yaml(work, Path(args.val_images_dir), args.names)
            print(f"[run] {args.protocol} ratio={ratio} seed={seed} | real={n_real} syn={n_syn}")
            row = run_one(
                task=args.task, model_name=args.model, data_yaml=data_yaml,
                protocol=args.protocol, seed=seed, epochs=args.epochs,
                imgsz=args.imgsz, project=project,
            )
            row.update({"ratio": ratio, "n_real": n_real, "n_syn": n_syn, "epochs": args.epochs, "imgsz": args.imgsz})
            all_rows.append(row)
            print(f"   -> {row}")

    df = pd.DataFrame(all_rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} per-run rows -> {out}")

    # Aggregate per (protocol, ratio) -> mean +/- std [bootstrap CI]
    print("\n" + "=" * 70)
    print("Aggregated (mean +/- std [bootstrap 95% CI])")
    print("=" * 70)
    summary_rows = []
    for ratio in ratios:
        sub = [r for r in all_rows if r.get("ratio") == ratio]
        if not sub:
            continue
        agg = aggregate_runs(sub, metric_keys)
        print(f"\n{args.protocol} ratio={ratio} (n={len(sub)} seeds):")
        for k, v in agg.items():
            print(f"  {k:>12}: {v['pretty']}")
        flat = {"protocol": args.protocol, "ratio": ratio, "n_runs": len(sub)}
        for k, v in agg.items():
            flat[f"{k}_mean"] = v["mean"]
            flat[f"{k}_std"] = v["std"]
            flat[f"{k}_ci_low"] = v["ci_low"]
            flat[f"{k}_ci_high"] = v["ci_high"]
        summary_rows.append(flat)

    summ_out = out.with_name(out.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summ_out, index=False)
    print(f"\nSaved aggregated summary -> {summ_out}")


if __name__ == "__main__":
    _cli()
