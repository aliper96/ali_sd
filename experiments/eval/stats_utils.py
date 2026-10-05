"""
stats_utils.py — Statistical reporting helpers for the Ali-AUG evaluation.

Addresses reviewer comment F (Pattern Recognition, R3 Q3 = "needs statistician"):
every reported metric should come with a measure of dispersion and, where the
sample is small, a confidence interval. These helpers are shared by the
generation-metric and downstream-evaluation scripts so that all tables report
mean +/- std and bootstrap 95% CIs in a consistent way.

No fabricated numbers: these functions only aggregate values you actually
measured over repeated runs / random splits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Sample mean and (population, ddof=0 if n==1 else ddof=1) std."""
    import numpy as np

    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return float("nan"), float("nan")
    ddof = 1 if arr.size > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for the mean.

    Returns (low, high) at the (1-alpha) level. With very few samples the CI is
    wide on purpose — that honesty is exactly what the reviewer asked for on the
    small-split experiments.
    """
    import numpy as np

    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def format_metric(values: Sequence[float], decimals: int = 3, with_ci: bool = True) -> str:
    """
    Human/LaTeX-friendly string: "mean +/- std [ci_low, ci_high]".

    Example: format_metric([0.81, 0.79, 0.83]) -> "0.810 +/- 0.020 [0.793, 0.827]"
    """
    m, s = mean_std(values)
    if m != m:
        return "n/a"
    base = f"{m:.{decimals}f} +/- {s:.{decimals}f}"
    if not with_ci:
        return base
    lo, hi = bootstrap_ci(values)
    return f"{base} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def latex_metric(values: Sequence[float], decimals: int = 3) -> str:
    """LaTeX cell: '$mean \\pm std$' (CIs go in a footnote/caption)."""
    m, s = mean_std(values)
    if m != m:
        return "n/a"
    return f"${m:.{decimals}f} \\pm {s:.{decimals}f}$"


def aggregate_runs(rows: Iterable[dict], metric_keys: List[str], decimals: int = 3) -> dict:
    """
    Aggregate a list of per-run metric dicts (e.g. one per seed/split) into a
    single summary dict with mean+/-std and bootstrap CI per metric.

    Each input row is {metric_key: value, ...}; output is
    {metric_key: {"mean", "std", "ci_low", "ci_high", "n", "pretty", "latex"}}.
    """
    rows = list(rows)
    summary: dict = {}
    for key in metric_keys:
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        m, s = mean_std(vals)
        lo, hi = bootstrap_ci(vals)
        summary[key] = {
            "mean": round(m, decimals) if m == m else None,
            "std": round(s, decimals) if s == s else None,
            "ci_low": round(lo, decimals) if lo == lo else None,
            "ci_high": round(hi, decimals) if hi == hi else None,
            "n": len(vals),
            "pretty": format_metric(vals, decimals=decimals),
            "latex": latex_metric(vals, decimals=decimals),
        }
    return summary
