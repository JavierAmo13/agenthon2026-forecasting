
"""Model ensembling by draw pooling.

Each member contributes ``w_i * n_draws`` of its joint draws; pooled rows are
exchangeable samples of the mixture distribution, so every final draw is a
complete joint scenario from ONE member — never a recombination of columns.

v2: weights are shrunk toward uniform, w* = (1 - gamma) w + gamma/M, with a
score floor before inversion — fewer all-or-nothing decisions built on a
handful of OOF residuals (plan 3.8).
"""

from __future__ import annotations

import numpy as np


def pool_draws(member_samples: list[np.ndarray], weights: list[float],
               n_draws: int, seed: int = 0) -> np.ndarray:
    """members: list of [m_i, n_assets, n_horizons]; returns [n_draws, ...]."""
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    counts = np.floor(w * n_draws).astype(int)
    counts[0] += n_draws - counts.sum()
    parts = []
    for samp, c in zip(member_samples, counts):
        if c <= 0:
            continue
        pick = rng.choice(samp.shape[0], size=c, replace=True)
        parts.append(samp[pick])
    out = np.concatenate(parts, axis=0)[:n_draws]
    rng.shuffle(out, axis=0)
    return out


def member_weights(scores: list[float], gamma: float = 0.25,
                   floor: float = 1e-6) -> list[float]:
    """Lower score -> higher weight (inverse), floored and shrunk to uniform."""
    s = np.clip(np.asarray(scores, dtype=float), floor, None)
    w = 1.0 / s
    w = w / w.sum()
    m = len(w)
    w = (1.0 - gamma) * w + gamma / m
    return w.tolist()
