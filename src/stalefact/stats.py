"""Bootstrap confidence intervals over conversations."""

from __future__ import annotations

import numpy as np

N_RESAMPLES = 1000
LEVEL = 0.95


def bootstrap_ratios(
    numerators: np.ndarray,
    denominators: np.ndarray,
    seed: int,
    n_resamples: int = N_RESAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ratio of sums per column with a percentile interval, resampling conversations.

    numerators has shape (conversations, k) and denominators shape (conversations,) or
    (conversations, k). Returns the point estimates and the lower and upper bounds, each of
    length k. The resample indices depend only on the seed and the number of conversations, so
    different cells are resampled with the same conversations.
    """
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if den.ndim == 1:
        den = den[:, None]
    n = num.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        point = num.sum(axis=0) / den.sum(axis=0)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_resamples, n))
        sums_n = num[idx].sum(axis=1)
        sums_d = np.broadcast_to(den, (n, num.shape[1]))[idx].sum(axis=1)
        ratios = sums_n / sums_d
    alpha = (1 - LEVEL) / 2
    lower = np.nanpercentile(ratios, 100 * alpha, axis=0)
    upper = np.nanpercentile(ratios, 100 * (1 - alpha), axis=0)
    return point, lower, upper
