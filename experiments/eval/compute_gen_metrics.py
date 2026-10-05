"""
compute_gen_metrics.py — Generation-quality metrics for Ali-AUG synthetics.

Computes the three metrics the paper reports for image generation:
  - FID      (clean-fid protocol; lower is better)
  - IS       (Inception Score; higher is better)
  - IC-LPIPS (intra-cluster LPIPS = mean pairwise LPIPS within a category;
              higher = more intra-class diversity)

It is backbone-agnostic: point it at a folder of generated images produced by
any generator and at the matching
real images. Run it once per backbone to fill the quality--cost trade-off table
(tab:backbone_tradeoff) — do NOT copy numbers between backbones.

Expected layout (two modes):

  Flat:                              Per-category (--per-category):
    real_dir/*.png                     real_dir/<class>/*.png
    gen_dir/*.png                      gen_dir/<class>/*.png

Usage:
    python eval/compute_gen_metrics.py \\
        --real-dir  /path/to/real \\
        --gen-dir   /path/to/generated \\
        --per-category \\
        --output-csv results/gen_metrics_other.csv \\
        --print-table

Dependencies (install only what you use):
    pip install clean-fid torchmetrics lpips torch torchvision pillow pandas numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ──────────────────────────────────────────────────────────────────────────────
# FID (clean-fid)
# ──────────────────────────────────────────────────────────────────────────────
def compute_fid(real_dir: Path, gen_dir: Path, device: str = "cuda") -> float:
    """clean-FID between two image folders. Matches the paper's clean-FID protocol."""
    try:
        from cleanfid import fid
    except ImportError:
        raise ImportError("clean-fid required: pip install clean-fid")
    return float(fid.compute_fid(str(real_dir), str(gen_dir), device=device))


# ──────────────────────────────────────────────────────────────────────────────
# Inception Score
# ──────────────────────────────────────────────────────────────────────────────
def compute_is(gen_dir: Path, device: str = "cuda", splits: int = 10) -> float:
    """Inception Score over the generated images (mean over splits)."""
    try:
        import torch
        from torchmetrics.image.inception import InceptionScore
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("torchmetrics+torch required: pip install torchmetrics torch torchvision")

    metric = InceptionScore(splits=splits, normalize=True).to(device)
    imgs = _list_images(gen_dir)
    if not imgs:
        return float("nan")
    batch = []
    for p in imgs:
        im = Image.open(p).convert("RGB").resize((299, 299))
        t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        batch.append(t)
        if len(batch) == 32:
            metric.update(torch.stack(batch).to(device))
            batch = []
    if batch:
        metric.update(torch.stack(batch).to(device))
    is_mean, _ = metric.compute()
    return float(is_mean)


# ──────────────────────────────────────────────────────────────────────────────
# IC-LPIPS (intra-cluster LPIPS): mean pairwise LPIPS within a category
# ──────────────────────────────────────────────────────────────────────────────
def compute_ic_lpips(
    gen_dir: Path,
    device: str = "cuda",
    max_pairs: int = 1000,
    seed: int = 0,
) -> float:
    """
    Mean pairwise LPIPS among generated images of one category.
    Samples up to max_pairs random pairs to bound cost on large sets.
    """
    try:
        import torch
        import lpips as lpips_lib
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("lpips+torch required: pip install lpips torch torchvision")

    imgs = _list_images(gen_dir)
    if len(imgs) < 2:
        return float("nan")

    loss_fn = lpips_lib.LPIPS(net="alex").to(device)

    def _load(p):
        im = Image.open(p).convert("RGB").resize((256, 256))
        t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        return (t * 2 - 1).unsqueeze(0).to(device)  # LPIPS expects [-1, 1]

    rng = np.random.default_rng(seed)
    n = len(imgs)
    # all pairs if small, else random sample
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    cache = {}
    def _get(i):
        if i not in cache:
            cache[i] = _load(imgs[i])
        return cache[i]

    dists = []
    with torch.no_grad():
        for i, j in pairs:
            dists.append(float(loss_fn(_get(i), _get(j)).item()))
    import numpy as _np
    return float(_np.mean(dists)) if dists else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# IC-LPIPS, STANDARD DEFINITION (Ojha et al., CVPR 2021) — use this one
# ──────────────────────────────────────────────────────────────────────────────
def compute_ic_lpips_clustered(
    gen_dir: Path,
    real_dir: Path,
    device: str = "cuda",
    max_pairs_per_cluster: int = 200,
    seed: int = 0,
) -> dict:
    """Intra-cluster LPIPS as defined by Ojha et al. (CVPR 2021) and used by
    AnomalyDiffusion / SeaS / MAGIC:

        1. assign every GENERATED image to its nearest REAL training sample
           (k clusters, one per real sample), by LPIPS distance;
        2. average pairwise LPIPS WITHIN each cluster;
        3. average over clusters.

    The clustering step is the whole point: it is what detects mode collapse.
    A model that memorises the k training defects and replays them scores ~0
    here, because every cluster is tight — while the un-clustered variant
    (`compute_ic_lpips`) would still report a large value, since different
    clusters differ from each other. The un-clustered number is therefore NOT
    comparable to published IC-LPIPS values; it is kept only for continuity
    with earlier internal runs.

    Returns dict with the metric, the number of non-empty clusters, and the
    per-cluster occupancy (a very unbalanced occupancy is itself a red flag).
    """
    import numpy as np
    import torch
    import lpips as lpips_lib
    from PIL import Image

    gen = _list_images(gen_dir)
    real = _list_images(real_dir)
    if len(gen) < 2 or not real:
        return {"IC_LPIPS": float("nan"), "n_clusters": 0, "occupancy": []}

    loss_fn = lpips_lib.LPIPS(net="alex").to(device)

    def _load(p):
        im = Image.open(p).convert("RGB").resize((256, 256))
        t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        return (t * 2 - 1).unsqueeze(0).to(device)

    real_t = [_load(p) for p in real]
    gen_t = [_load(p) for p in gen]

    # 1. assign each generated image to its nearest real sample
    assign = []
    with torch.no_grad():
        for g in gen_t:
            d = [float(loss_fn(g, r).item()) for r in real_t]
            assign.append(int(np.argmin(d)))
    assign = np.asarray(assign)

    # 2. mean pairwise LPIPS inside each cluster
    rng = np.random.default_rng(seed)
    per_cluster, occupancy = [], []
    with torch.no_grad():
        for k in range(len(real_t)):
            idx = np.where(assign == k)[0]
            occupancy.append(int(idx.size))
            if idx.size < 2:
                continue                      # singleton clusters contribute nothing
            pairs = [(int(a), int(b)) for i, a in enumerate(idx) for b in idx[i + 1:]]
            if len(pairs) > max_pairs_per_cluster:
                sel = rng.choice(len(pairs), size=max_pairs_per_cluster, replace=False)
                pairs = [pairs[s] for s in sel]
            d = [float(loss_fn(gen_t[a], gen_t[b]).item()) for a, b in pairs]
            per_cluster.append(float(np.mean(d)))

    # 3. average over non-empty clusters
    return {
        "IC_LPIPS": float(np.mean(per_cluster)) if per_cluster else float("nan"),
        "n_clusters": len(per_cluster),
        "occupancy": occupancy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(
    real_dir: Path,
    gen_dir: Path,
    per_category: bool,
    categories: Optional[List[str]],
    device: str,
    do_fid: bool,
    do_is: bool,
    do_iclpips: bool,
) -> "pd.DataFrame":
    import pandas as pd

    if per_category:
        if categories:
            cats = categories
        else:
            cats = sorted(d.name for d in gen_dir.iterdir() if d.is_dir())
    else:
        cats = ["all"]

    rows = []
    for cat in cats:
        rdir = real_dir / cat if per_category else real_dir
        gdir = gen_dir / cat if per_category else gen_dir
        if not gdir.exists():
            print(f"[skip] {cat}: no generated dir at {gdir}")
            continue

        row = {"category": cat, "n_gen": len(_list_images(gdir))}
        if do_fid:
            try:
                row["FID"] = round(compute_fid(rdir, gdir, device), 3)
            except Exception as e:
                print(f"[FID fail] {cat}: {e}")
                row["FID"] = None
        if do_is:
            try:
                row["IS"] = round(compute_is(gdir, device), 3)
            except Exception as e:
                print(f"[IS fail] {cat}: {e}")
                row["IS"] = None
        if do_iclpips:
            # Standard clustered IC-LPIPS (Ojha et al.) — the comparable number.
            try:
                res = compute_ic_lpips_clustered(gdir, rdir, device)
                row["IC_LPIPS"] = round(res["IC_LPIPS"], 3)
                row["n_clusters"] = res["n_clusters"]
                occ = res["occupancy"]
                # concentration of generated images in a single cluster: a value
                # near 1.0 means the model keeps reproducing one training defect
                row["max_cluster_frac"] = (
                    round(max(occ) / sum(occ), 3) if occ and sum(occ) else None
                )
            except Exception as e:
                print(f"[IC-LPIPS fail] {cat}: {e}")
                row["IC_LPIPS"] = None
            # Legacy un-clustered variant, reported only for continuity with
            # earlier internal runs. NOT comparable to published IC-LPIPS.
            try:
                row["IC_LPIPS_unclustered"] = round(compute_ic_lpips(gdir, device), 3)
            except Exception:
                row["IC_LPIPS_unclustered"] = None
        rows.append(row)
        print(f"[done] {cat}: {row}")

    df = pd.DataFrame(rows)
    # Average row across categories (unweighted, matching the paper's "Average" row)
    if per_category and not df.empty:
        avg = {"category": "Average", "n_gen": int(df["n_gen"].sum())}
        for m in ("FID", "IS", "IC_LPIPS", "IC_LPIPS_unclustered", "max_cluster_frac"):
            if m in df.columns:
                avg[m] = round(df[m].dropna().mean(), 3) if df[m].notna().any() else None
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df


def _cli():
    p = argparse.ArgumentParser(description="FID / IS / IC-LPIPS for Ali-AUG synthetics.")
    p.add_argument("--real-dir", required=True)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--per-category", action="store_true",
                   help="real-dir/<class> and gen-dir/<class> subfolders.")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Explicit category list (default: all subdirs of gen-dir).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--no-is", action="store_true")
    p.add_argument("--no-iclpips", action="store_true")
    p.add_argument("--output-csv", default="results/gen_metrics.csv")
    p.add_argument("--print-table", action="store_true")
    args = p.parse_args()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = evaluate(
        real_dir=Path(args.real_dir),
        gen_dir=Path(args.gen_dir),
        per_category=args.per_category,
        categories=args.categories,
        device=args.device,
        do_fid=not args.no_fid,
        do_is=not args.no_is,
        do_iclpips=not args.no_iclpips,
    )
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    if args.print_table:
        print(df.to_string(index=False))


if __name__ == "__main__":
    _cli()
