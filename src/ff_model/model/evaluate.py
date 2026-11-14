"""Out-of-sample checks for the marginals: PIT calibration and pinball loss vs baselines."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..research.projection_errors import RELEVANT_TOP_N, build, relevant
from .marginals import TAUS, Marginals, cdf, pinball


def baseline_shift(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """proj + this position's residual quantiles."""
    q = np.zeros((test.height, len(TAUS)))
    for pos in RELEVANT_TOP_N:
        res = train.filter(pl.col("pos") == pos)["err"].to_numpy()
        idx = np.flatnonzero(test["pos"].to_numpy() == pos)
        q[idx] = test["proj"].to_numpy()[idx, None] + np.quantile(res, TAUS)[None, :]
    return q


def baseline_binned(train: pl.DataFrame, test: pl.DataFrame, n_bins: int = 6) -> np.ndarray:
    """proj + residual quantiles from players with a similar projection (same position)."""
    q = np.zeros((test.height, len(TAUS)))
    for pos in RELEVANT_TOP_N:
        tr = train.filter(pl.col("pos") == pos)
        edges = np.quantile(tr["proj"].to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1])
        tb = np.digitize(tr["proj"].to_numpy(), edges)
        idx = np.flatnonzero(test["pos"].to_numpy() == pos)
        sb = np.digitize(test["proj"].to_numpy()[idx], edges)
        for b in range(n_bins):
            res = tr["err"].to_numpy()[tb == b]
            sel = idx[sb == b]
            q[sel] = test["proj"].to_numpy()[sel, None] + np.quantile(res, TAUS)[None, :]
    return q


def report(train_seasons, test_seasons, depth: float = 2.0):
    all_df = build(sorted(set(train_seasons) | set(test_seasons)))
    train = relevant(all_df.filter(pl.col("season").is_in(train_seasons)), depth=depth)
    test = relevant(all_df.filter(pl.col("season").is_in(test_seasons)))
    model = Marginals().fit(train)
    qm, q1, q2 = model.quantiles(test), baseline_shift(train, test), baseline_binned(train, test)
    y = test["actual"].to_numpy()
    rows = []
    rng = np.random.default_rng(0)
    for pos in RELEVANT_TOP_N:
        i = np.flatnonzero(test["pos"].to_numpy() == pos)
        u = cdf(qm[i], y[i], rng)
        hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(i)
        rows.append({
            "pos": pos, "n": len(i),
            "pinball_model": pinball(qm[i], y[i]), "pinball_shift": pinball(q1[i], y[i]),
            "pinball_binned": pinball(q2[i], y[i]),
            "below_p10": float((u < 0.1).mean()), "above_p90": float((u > 0.9).mean()),
            "in_p25_p75": float(((u > 0.25) & (u < 0.75)).mean()),
            "pit_max_dev": float(np.abs(hist - 0.1).max()),
            "pit_deciles": " ".join(f"{h*100:.0f}" for h in hist),
        })
    return model, test, qm, pl.DataFrame(rows)
