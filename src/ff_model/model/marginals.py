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

from ..config import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "marginals.json"

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


# Pre-game context that the factor screen found ESPN under-adjusts for. Each enters
# as proj x context, so effects scale with the player's volume, and differs by tau,
# so it can move the floor and ceiling differently.
def context(df: pl.DataFrame) -> pl.DataFrame:
    def col(name, default=None):
        return pl.col(name) if name in df.columns else pl.lit(default)
    outdoor = col("roof").is_in(["outdoors", "open"]).fill_null(False)
    status = col("report_status").fill_null("none")
    practice = col("practice_status").fill_null("")
    listed = status.is_in(["Questionable", "Doubtful"])
    return df.with_columns(
        (~outdoor & col("roof").is_not_null()).cast(pl.Float64).alias("c_dome"),
        pl.when(outdoor).then(col("wind").fill_null(0) / 10).otherwise(0.0).alias("c_wind"),
        pl.when(outdoor).then((50 - col("temp").fill_null(60)).clip(0, None) / 10).otherwise(0.0).alias("c_cold"),
        (col("exp_margin").fill_null(0) / 7).clip(-2, 2).alias("c_margin"),
        (listed & practice.str.contains("Did Not")).cast(pl.Float64).alias("c_q_dnp"),
        (listed & practice.str.contains("Limited")).cast(pl.Float64).alias("c_q_limited"),
        (listed & ~practice.str.contains("Did Not|Limited")).cast(pl.Float64).alias("c_q_other"),
        col("games_back", 0).fill_null(0).is_between(1, 2).cast(pl.Float64).alias("c_return"),
        # starters out per the injury report (Out/Doubtful/IR), capped at 3
        col("ol_out", 0).fill_null(0).clip(0, 3).cast(pl.Float64).alias("c_ol_out"),
        col("opp_ol_out", 0).fill_null(0).clip(0, 3).cast(pl.Float64).alias("c_opp_ol_out"),
        col("opp_db_out", 0).fill_null(0).clip(0, 3).cast(pl.Float64).alias("c_opp_db_out"),
    )


WEATHER = ["c_dome", "c_wind", "c_cold"]
INJURY = ["c_q_dnp", "c_q_limited", "c_q_other", "c_return"]
# QB injury terms dropped: ~80 listed-QB weeks in training, and they hurt out of sample.
CONTEXT = {"QB": WEATHER + ["c_margin", "c_ol_out", "c_opp_db_out"], "RB": ["c_margin", "c_ol_out"] + INJURY,
           "WR": WEATHER + ["c_margin"] + INJURY, "TE": WEATHER + ["c_margin"] + INJURY,
           "K": WEATHER + ["c_margin"], "D/ST": ["c_margin", "c_opp_ol_out"]}

# design columns per position: proj plus proj x share interactions
SHARES = {"QB": ["f_pass_td_share", "f_rush_share"],
          "RB": ["f_td_share", "f_rec_share"], "WR": ["f_td_share", "f_rec_share"],
          "TE": ["f_td_share", "f_rec_share"], "K": [], "D/ST": []}


# K and D/ST projections are compressed and weakly predictive; a curved term
# overfits and made kicker ceilings fall as projections rose. Keep them linear.
CURVED = {"QB", "RB", "WR", "TE"}


def _design(df: pl.DataFrame, pos: str, context_cols: list[str], use_shares: bool = True) -> np.ndarray:
    # sqrt(proj) lets the spread bend with projection size instead of growing linearly
    proj = df["proj"].to_numpy()
    cols = [proj] + ([np.sqrt(np.clip(proj, 0, None))] if pos in CURVED else [])
    cols += [proj * df[c].to_numpy() for c in SHARES[pos]] if use_shares else []
    cols += [proj * df[c].to_numpy() for c in context_cols]
    return np.column_stack(cols)


@dataclass
class Marginals:
    coefs: dict[str, list[list[float]]] = field(default_factory=dict)  # pos -> per tau [intercept, *betas]
    use_context: bool = False
    context_cols: dict[str, list[str]] = field(default_factory=dict)  # frozen at fit time
    use_shares: bool = True  # False = "generic" shape: position + projection size only

    def _ctx(self, pos: str) -> list[str]:
        return self.context_cols.get(pos, []) if self.use_context else []

    def _prep(self, df: pl.DataFrame) -> pl.DataFrame:
        df = features(df)
        return context(df) if self.use_context else df

    def fit(self, train: pl.DataFrame, alpha: float = 0.0) -> "Marginals":
        train = self._prep(train)
        self.context_cols = {p: list(CONTEXT[p]) for p in SHARES} if self.use_context else {}
        for pos in SHARES:
            d = train.filter(pl.col("pos") == pos)
            X, y = _design(d, pos, self._ctx(pos), self.use_shares), d["actual"].to_numpy()
            rows = []
            for tau in TAUS:
                m = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs").fit(X, y)
                rows.append([float(m.intercept_), *map(float, m.coef_)])
            self.coefs[pos] = rows
        return self

    def quantiles(self, df: pl.DataFrame) -> np.ndarray:
        """(n, len(TAUS)) predicted quantiles, monotone in tau."""
        df = self._prep(df)
        out = np.zeros((df.height, len(TAUS)))
        pos = df["pos"].to_numpy()
        for p, rows in self.coefs.items():
            idx = np.flatnonzero(pos == p)
            if idx.size == 0:
                continue
            X = _design(df[idx], p, self._ctx(p), self.use_shares)
            B = np.array(rows)  # (tau, 1 + k)
            out[idx] = B[:, 0] + X @ B[:, 1:].T
        return np.maximum.accumulate(out, axis=1)  # no crossing

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"taus": TAUS.tolist(), "use_context": self.use_context, "use_shares": self.use_shares,
                                    "context_cols": self.context_cols, "coefs": self.coefs}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Marginals":
        d = json.loads(path.read_text())
        return cls(coefs=d["coefs"], use_context=d.get("use_context", False),
                   context_cols=d.get("context_cols", {}), use_shares=d.get("use_shares", True))


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


def fit_live(seasons=range(2019, 2026), depth: float = 2.0) -> Marginals:
    """Fit on every available season (active players, with context) and save to MODEL_PATH.

    Live inputs needed per player: weather at the stadium (roof, wind, temp), the
    team's expected margin from the archived odds, injury report status and final
    practice status, and games_back after an absence.
    """
    from ..research.factor_tests import factor_frame
    train = factor_frame(seasons, depth=depth, with_u=False).filter(pl.col("played"))
    m = Marginals(use_context=True).fit(train)
    m.save(MODEL_PATH)
    return m


def curve_mean(q: np.ndarray) -> np.ndarray:
    """Mean of each row's distribution (quantile curve + the same linear tails the simulator uses)."""
    grid = np.linspace(0.0005, 0.9995, 1000)
    t = np.concatenate([[0.0], TAUS, [1.0]])
    lo_w = np.maximum(q[:, 1] - q[:, 0], 1e-6) * TAUS[0] / (TAUS[1] - TAUS[0])
    hi_w = np.maximum(q[:, -1] - q[:, -2], 1e-6) * (1 - TAUS[-1]) / (TAUS[-1] - TAUS[-2])
    full = np.column_stack([q[:, 0] - lo_w, q, q[:, -1] + hi_w])
    return np.array([np.interp(grid, t, row).mean() for row in full])


def anchor(q: np.ndarray, proj: np.ndarray, q_base: np.ndarray | None = None) -> np.ndarray:
    """Shift each curve so its mean is ESPN's projection (plus, if q_base is given, the
    context's own shift of the mean: mean(q) - mean(q_base)). Shape and spread unchanged."""
    target = proj + ((curve_mean(q) - curve_mean(q_base)) if q_base is not None else 0.0)
    return q + (target - curve_mean(q))[:, None]


# Context inputs set to "no information" (used to measure what the context adds).
CONTEXT_NEUTRAL = {"roof": None, "wind": None, "temp": None, "exp_margin": None, "report_status": "none",
                   "practice_status": None, "games_back": 0, "ol_out": 0, "opp_ol_out": 0, "opp_db_out": 0}


def neutralize(df: pl.DataFrame, cols: dict | None = None) -> pl.DataFrame:
    cols = CONTEXT_NEUTRAL if cols is None else cols
    return df.with_columns([pl.lit(v).cast(df.schema[c]).alias(c) for c, v in cols.items() if c in df.columns])


def espn_context_quantiles(m: "Marginals", df: pl.DataFrame) -> np.ndarray:
    """Live distributions ("ESPN + context", the best variant in the walk-forward backtest):
    the model's shape, shifted so the mean = ESPN's projection + the context's own shift."""
    q = m.quantiles(df)
    return anchor(q, df["proj"].to_numpy(), m.quantiles(neutralize(df)))
