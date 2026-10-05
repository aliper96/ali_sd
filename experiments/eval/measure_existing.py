"""
measure_existing.py — clustered IC-LPIPS (+ FID) on the surviving Img2Img-turbo
generations found at E:\\PycharmProjects\\aliaug2\\data\\mvtec_ad.

Cluster centres are the REAL anomalous training images (`train_B`), which is the
closest available analogue of Ojha et al.'s protocol (generated images assigned to
their nearest real training sample). Occupancy is reported alongside the metric:
with only a few hundred generated images per category the clusters are thin, and a
metric computed over mostly-singleton clusters is not trustworthy. The standard
protocol generates ~1000+ images per category.

Run:
  python eval/measure_existing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gen_metrics import compute_ic_lpips, compute_ic_lpips_clustered  # noqa: E402

BASE = Path(r"E:\PycharmProjects\aliaug2\data\mvtec_ad")
CATS = ["bottle", "carpet", "tile", "wood"]
DEV = "cuda"


def fid(real: Path, gen: Path):
    try:
        from cleanfid import fid as _f
        return round(_f.compute_fid(str(real), str(gen), device=DEV), 2)
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def main():
    rows = []
    for cat in CATS:
        gi = BASE / cat / "output" / "generated_images"
        centers = BASE / cat / "train_B"          # real anomalous training images
        target = gi / "target"
        for variant in ("predicted", "predicted_pld"):
            gdir = gi / variant
            n = len(list(gdir.glob("*.png"))) + len(list(gdir.glob("*.jpg"))) if gdir.exists() else 0
            if n == 0:
                continue
            print(f"\n=== {cat} / {variant}  ({n} images) ===", flush=True)
            res = compute_ic_lpips_clustered(gdir, centers, device=DEV)
            old = compute_ic_lpips(gdir, device=DEV)
            occ = res["occupancy"]
            nonempty = sum(1 for o in occ if o >= 2)
            rows.append({
                "cat": cat, "variant": variant, "n_gen": n,
                "n_centers": len(occ),
                "IC_LPIPS": round(res["IC_LPIPS"], 4),
                "clusters_used": res["n_clusters"],
                "clusters_ge2": nonempty,
                "max_frac": round(max(occ) / sum(occ), 3) if occ and sum(occ) else None,
                "IC_LPIPS_old": round(old, 4),
                "FID_vs_target": fid(target, gdir) if target.exists() else "n/a",
            })
            print(f"  IC-LPIPS (clustered) = {rows[-1]['IC_LPIPS']}   "
                  f"[{rows[-1]['clusters_used']}/{rows[-1]['n_centers']} clusters usable]")
            print(f"  IC-LPIPS (old)       = {rows[-1]['IC_LPIPS_old']}")
            print(f"  FID vs target        = {rows[-1]['FID_vs_target']}")

    print("\n" + "=" * 100)
    hdr = f"{'cat':<9}{'variant':<16}{'n_gen':>7}{'IC-LPIPS':>10}{'clusters':>10}{'max_frac':>10}{'IC old':>9}{'FID':>10}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['cat']:<9}{r['variant']:<16}{r['n_gen']:>7}{r['IC_LPIPS']:>10}"
              f"{str(r['clusters_used'])+'/'+str(r['n_centers']):>10}{str(r['max_frac']):>10}"
              f"{r['IC_LPIPS_old']:>9}{str(r['FID_vs_target']):>10}")
    print("=" * 100)
    print("\nIC-LPIPS higher = more diverse. FID lower = closer to the real distribution.")
    print("Read them TOGETHER: diversity alone is trivially maximised by noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
