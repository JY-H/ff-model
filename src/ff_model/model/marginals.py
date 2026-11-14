"""Per-player outcome distributions (marginals), as conditional quantiles.

For each position and quantile level tau, a linear quantile regression predicts
actual league points from the ESPN projection and the projection's make-up
(TD share, reception share, QB rushing share), all interacted with the
projection so the spread scales with it. A player's distribution is the
resulting quantile curve; the CDF interpolates between levels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import QuantileRegressor

TAUS = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])


def features(df: pl.DataFrame) -> pl.DataFrame:
    """Shares of the projection, clipped to [0, 1]. Zero for K/D-ST."""
    p = pl.col("proj").clip(lower_bound=0.5)
    td_pts = 6 * (pl.col("p_rush_td") + pl.col("p_rec_td"))
    return df.with_columns(
        (td_pts / p).clip(0, 1).alias("f_td_share"),
        (pl.col("p_rec") / p).clip(0, 1).alias("f_rec_share"),
        (4 * pl.col("p_pass_td") / p).clip(0, 1).alias("f_pass_td_share"),
        ((0.1 * pl.col("p_rush_yds") + 6 * pl.col("p_rush_td")) / p).clip(0, 1).alias("f_rush_share"),
    )


# design columns per position: proj plus proj x share interactions
SHARES = {"QB": ["f_pass_td_share", "f_rush_share"],
          "RB": ["f_td_share", "f_rec_share"], "WR": ["f_td_share", "f_rec_share"],
          "TE": ["f_td_share", "f_rec_share"], "K": [], "D/ST": []}


def _design(df: pl.DataFrame, pos: str) -> np.ndarray:
    # sqrt(proj) lets the spread bend with projection size instead of growing linearly
    proj = df["proj"].to_numpy()
    cols = [proj, np.sqrt(np.clip(proj, 0, None))] + [proj * df[c].to_numpy() for c in SHARES[pos]]
    return np.column_stack(cols)


@dataclass
class Marginals:
    coefs: dict[str, list[list[float]]] = field(default_factory=dict)  # pos -> per tau [intercept, *betas]

    def fit(self, train: pl.DataFrame, alpha: float = 0.0) -> "Marginals":
        train = features(train)
        for pos in SHARES:
            d = train.filter(pl.col("pos") == pos)
            X, y = _design(d, pos), d["actual"].to_numpy()
            rows = []
            for tau in TAUS:
                m = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs").fit(X, y)
                rows.append([float(m.intercept_), *map(float, m.coef_)])
            self.coefs[pos] = rows
        return self

    def quantiles(self, df: pl.DataFrame) -> np.ndarray:
        """(n, len(TAUS)) predicted quantiles, monotone in tau."""
        df = features(df)
        out = np.zeros((df.height, len(TAUS)))
        pos = df["pos"].to_numpy()
        for p, rows in self.coefs.items():
            idx = np.flatnonzero(pos == p)
            if idx.size == 0:
                continue
            X = _design(df[idx], p)
            B = np.array(rows)  # (tau, 1 + k)
            out[idx] = B[:, 0] + X @ B[:, 1:].T
        return np.maximum.accumulate(out, axis=1)  # no crossing

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"taus": TAUS.tolist(), "coefs": self.coefs}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Marginals":
        return cls(coefs=json.loads(path.read_text())["coefs"])


def cdf(q: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomized PIT: F(y) from each row's quantile curve.

    Linear between quantile levels, linear tails to 0/1 beyond the end levels
    (tail widths taken from the nearest interval). Where y sits on a flat
    stretch (a point mass, e.g. 0 points), u is drawn uniformly across it.
    """
    rng = rng or np.random.default_rng(0)
    n, k = q.shape
    t = np.concatenate([[0.0], TAUS, [1.0]])
    lo_w = np.maximum(q[:, 1] - q[:, 0], 1e-6) * TAUS[0] / (TAUS[1] - TAUS[0])
    hi_w = np.maximum(q[:, -1] - q[:, -2], 1e-6) * (1 - TAUS[-1]) / (TAUS[-1] - TAUS[-2])
    full = np.column_stack([q[:, 0] - lo_w, q, q[:, -1] + hi_w])
    # A tiny ramp makes each row strictly increasing; probing at y -/+ eps (eps >> ramp)
    # then brackets any flat stretch, so a point mass at y gets the whole span.
    full = full + 1e-9 * np.arange(full.shape[1])
    eps = 1e-6
    u = np.empty(n)
    for i in range(n):
        lo = np.interp(y[i] - eps, full[i], t, left=0.0, right=1.0)
        hi = np.interp(y[i] + eps, full[i], t, left=0.0, right=1.0)
        u[i] = rng.uniform(lo, hi) if hi > lo + 1e-6 else (lo + hi) / 2
    return u


def pinball(q: np.ndarray, y: np.ndarray) -> float:
    """Mean pinball loss across TAUS (~ CRPS / 2). Lower is better."""
    d = y[:, None] - q
    return float(np.mean(np.maximum(TAUS * d, (TAUS - 1) * d)))
