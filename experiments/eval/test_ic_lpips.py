"""
test_ic_lpips.py — sanity check for the clustered IC-LPIPS implementation.

Pins the property that motivates the metric: a generator that MEMORISES the
training samples must score ~0, while a diverse generator must score clearly
higher. The previously used un-clustered variant fails exactly this test, which
is why it is not comparable to published IC-LPIPS numbers.

Run:
  python eval/test_ic_lpips.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gen_metrics import compute_ic_lpips, compute_ic_lpips_clustered  # noqa: E402

DEV = "cuda"
SIZE = 128


def _write(path: Path, arr: np.ndarray):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _base(seed: int) -> np.ndarray:
    """A distinctive 'real training sample': coloured blocks + structure."""
    rng = np.random.default_rng(seed)
    img = np.zeros((SIZE, SIZE, 3), np.float64)
    img[:, :] = rng.integers(40, 210, 3)
    for _ in range(6):
        y, x = rng.integers(0, SIZE - 30, 2)
        img[y:y + 30, x:x + 30] = rng.integers(0, 255, 3)
    return img


def main():
    tmp = Path(tempfile.mkdtemp())
    real = tmp / "real"; real.mkdir()
    memo = tmp / "gen_memorised"; memo.mkdir()
    div = tmp / "gen_diverse"; div.mkdir()

    K = 3                      # three real training samples -> three clusters
    bases = [_base(s) for s in range(K)]
    for i, b in enumerate(bases):
        _write(real / f"real_{i}.png", b)

    rng = np.random.default_rng(123)
    # (a) MEMORISING generator: replays each real sample almost exactly
    for i, b in enumerate(bases):
        for j in range(6):
            _write(memo / f"g_{i}_{j}.png", np.clip(b + rng.normal(0, 1.0, b.shape), 0, 255))
    # (b) DIVERSE generator: same modes, but genuine within-mode variation
    for i, b in enumerate(bases):
        for j in range(6):
            v = b.copy()
            y, x = rng.integers(0, SIZE - 40, 2)
            v[y:y + 40, x:x + 40] = rng.integers(0, 255, 3)
            v = np.clip(v + rng.normal(0, 18.0, v.shape), 0, 255)
            _write(div / f"g_{i}_{j}.png", v)

    r_memo = compute_ic_lpips_clustered(memo, real, device=DEV)
    r_div = compute_ic_lpips_clustered(div, real, device=DEV)
    u_memo = compute_ic_lpips(memo, device=DEV)
    u_div = compute_ic_lpips(div, device=DEV)

    print(f"{'':22}{'clustered (correct)':>22}{'un-clustered (old)':>22}")
    print(f"{'memorising generator':22}{r_memo['IC_LPIPS']:>22.4f}{u_memo:>22.4f}")
    print(f"{'diverse generator':22}{r_div['IC_LPIPS']:>22.4f}{u_div:>22.4f}")
    print(f"\nclusters found: memorising={r_memo['n_clusters']} diverse={r_div['n_clusters']}")
    print(f"occupancy     : memorising={r_memo['occupancy']} diverse={r_div['occupancy']}")

    ok = []
    ok.append(("memorising scores near zero", r_memo["IC_LPIPS"] < 0.05))
    ok.append(("diverse scores clearly higher", r_div["IC_LPIPS"] > 4 * max(r_memo["IC_LPIPS"], 1e-6)))
    ok.append(("all clusters populated", r_div["n_clusters"] == K))
    # the old metric cannot separate the two cases nearly as sharply
    sep_new = r_div["IC_LPIPS"] / max(r_memo["IC_LPIPS"], 1e-6)
    sep_old = u_div / max(u_memo, 1e-6)
    ok.append(("clustered separates better than un-clustered", sep_new > sep_old))

    print()
    for name, cond in ok:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    print(f"\nseparation ratio  clustered={sep_new:.1f}x  un-clustered={sep_old:.1f}x")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all(c for _, c in ok) else 1


if __name__ == "__main__":
    sys.exit(main())
