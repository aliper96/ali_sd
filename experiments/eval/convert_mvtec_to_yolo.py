"""
convert_mvtec_to_yolo.py — MVTec-AD (and Ali-AUG synthetics) -> YOLO-seg format.

Builds the YOLO datasets consumed by run_downstream_yolo.py. Two modes:

  --mode mvtec   : real data from a MVTec-AD category directory, e.g.
                     mvtec_anomaly_detection/tile/
                       test/<class>/*.png
                       ground_truth/<class>/<stem>_mask.png
                   Produces a fixed train/val split of REAL defect images with
                   polygon labels derived from the ground-truth masks. The val
                   split is the held-out real set used for ALL protocols.

  --mode synthetic : Ali-AUG generations from either backbone, given as
                       <syn-dir>/<class>/<stem>.png        (generated image)
                       <syn-dir>/<class>/<stem>_mask.png   (input mask used)
                     OR a flat dir with class taken from the filename prefix and
                     masks in --mask-dir. Produces images/ + labels/ (no split;
                     synthetics only ever go into the training set).

Class order is fixed and shared with the generator recipe (tile defects):
    crack glue_strip gray_stroke oil
Override with --names for other MVTec categories.

Mask -> polygon uses OpenCV contours (one polygon per connected component),
normalised to [0,1]; this is the standard YOLO-seg label format:
    <cls> x1 y1 x2 y2 ... xn yn

Usage:
    # Real tile, 70/30 split, seed 0
    python eval/convert_mvtec_to_yolo.py --mode mvtec \\
        --category-dir mvtec_anomaly_detection/tile \\
        --names crack glue_strip gray_stroke oil \\
        --val-frac 0.30 --seed 0 \\
        --out results/data/real_tile

    # Synthetic generations (per-class subfolders with *_mask.png)
    python eval/convert_mvtec_to_yolo.py --mode synthetic \\
        --syn-dir results/gen/other_generator_tile \\
        --names crack glue_strip gray_stroke oil \\
        --out results/data/syn_tile

Dependencies:
    pip install opencv-python pillow numpy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_NAMES = ["crack", "glue_strip", "gray_stroke", "oil"]
MIN_CONTOUR_AREA_PX = 20  # drop specks; mirrors the generator's min-mask-area filter


def _imgs(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def mask_to_polygons(mask_path: Path, min_area: int = MIN_CONTOUR_AREA_PX):
    """Return list of normalised polygons [[x1,y1,...], ...] from a binary mask."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("opencv required: pip install opencv-python numpy")

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    h, w = m.shape[:2]
    _, binm = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        # simplify the contour a bit
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        norm = []
        for x, y in approx:
            norm += [x / w, y / h]
        polys.append(norm)
    return polys


def write_label(label_path: Path, cls_id: int, polys: List[List[float]]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for poly in polys:
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{cls_id} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def find_mask_for(stem: str, gt_dir: Path) -> Optional[Path]:
    """MVTec masks are named <stem>_mask.png; also accept <stem>.png."""
    for cand in (gt_dir / f"{stem}_mask.png", gt_dir / f"{stem}.png", gt_dir / f"{stem}_mask.jpg"):
        if cand.exists():
            return cand
    return None


def convert_mvtec(category_dir: Path, names: List[str], val_frac: float, seed: int, out: Path):
    import numpy as np

    name_to_id = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(seed)

    splits = {"train": [], "val": []}
    for cls in names:
        test_dir = category_dir / "test" / cls
        gt_dir = category_dir / "ground_truth" / cls
        if not test_dir.exists():
            print(f"[skip] no test images for class {cls}")
            continue
        items = _imgs(test_dir)
        rng.shuffle(items)
        n_val = int(round(val_frac * len(items)))
        val_set = set(items[:n_val])
        for img in items:
            mask = find_mask_for(img.stem, gt_dir)
            split = "val" if img in val_set else "train"
            splits[split].append((cls, img, mask))

    counts = {}
    for split, entries in splits.items():
        for cls, img, mask in entries:
            dst_img = out / split / "images" / f"{cls}_{img.name}"
            dst_lbl = out / split / "labels" / f"{cls}_{img.stem}.txt"
            _link_or_copy(img, dst_img)
            polys = mask_to_polygons(mask) if mask else []
            write_label(dst_lbl, name_to_id[cls], polys)
            counts[(split, cls)] = counts.get((split, cls), 0) + 1

    print("\nMVTec -> YOLO-seg summary:")
    for split in ("train", "val"):
        total = sum(v for (s, _), v in counts.items() if s == split)
        per = {c: counts.get((split, c), 0) for c in names}
        print(f"  {split}: {total}  {per}")
    print(f"  output: {out.resolve()}  (train/images, train/labels, val/images, val/labels)")
    print("  NOTE: val is the held-out REAL split — never add synthetics to it.")


def convert_synthetic(syn_dir: Path, names: List[str], mask_dir: Optional[Path], out: Path):
    name_to_id = {n: i for i, n in enumerate(names)}
    per_cls_dirs = [d for d in syn_dir.iterdir() if d.is_dir() and d.name in name_to_id] if syn_dir.is_dir() else []

    n = 0
    if per_cls_dirs:  # per-class subfolders, masks as <stem>_mask.png alongside
        for cdir in per_cls_dirs:
            cls = cdir.name
            for img in _imgs(cdir):
                if img.stem.endswith("_mask"):
                    continue
                mask = cdir / f"{img.stem}_mask.png"
                polys = mask_to_polygons(mask) if mask.exists() else []
                _link_or_copy(img, out / "images" / f"{cls}_{img.name}")
                write_label(out / "labels" / f"{cls}_{img.stem}.txt", name_to_id[cls], polys)
                n += 1
    else:  # flat dir; class from filename prefix, masks in --mask-dir
        for img in _imgs(syn_dir):
            if img.stem.endswith("_mask"):
                continue
            cls = next((c for c in names if img.stem.startswith(c)), None)
            if cls is None:
                print(f"[skip] {img.name}: class prefix not in {names}")
                continue
            mask = None
            if mask_dir:
                for cand in (mask_dir / f"{img.stem}_mask.png", mask_dir / f"{img.stem}.png"):
                    if cand.exists():
                        mask = cand
                        break
            polys = mask_to_polygons(mask) if mask else []
            _link_or_copy(img, out / "images" / img.name)
            write_label(out / "labels" / f"{img.stem}.txt", name_to_id[cls], polys)
            n += 1

    print(f"\nSynthetic -> YOLO-seg: {n} images -> {out.resolve()} (images/, labels/)")


def _cli():
    p = argparse.ArgumentParser(description="MVTec-AD / Ali-AUG synthetics -> YOLO-seg labels.")
    p.add_argument("--mode", choices=["mvtec", "synthetic"], required=True)
    p.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    p.add_argument("--out", required=True)
    # mvtec
    p.add_argument("--category-dir", default=None, help="(mvtec) e.g. mvtec_anomaly_detection/tile")
    p.add_argument("--val-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    # synthetic
    p.add_argument("--syn-dir", default=None, help="(synthetic) generated images dir")
    p.add_argument("--mask-dir", default=None, help="(synthetic, flat mode) masks dir")
    args = p.parse_args()

    out = Path(args.out)
    if args.mode == "mvtec":
        if not args.category_dir:
            p.error("--category-dir required for --mode mvtec")
        convert_mvtec(Path(args.category_dir), args.names, args.val_frac, args.seed, out)
    else:
        if not args.syn_dir:
            p.error("--syn-dir required for --mode synthetic")
        convert_synthetic(Path(args.syn_dir), args.names,
                          Path(args.mask_dir) if args.mask_dir else None, out)


if __name__ == "__main__":
    _cli()
