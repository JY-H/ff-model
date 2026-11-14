"""How players' outcomes move together: a role-pair Gaussian copula.

Each player-week's result is expressed as where it landed in his own predicted
range (PIT u), then as a normal score z = Phi^-1(u). Within a game, players get
a role by ESPN projection rank (QB1, RB1, RB2, WR1-3, TE1, K, DST). We estimate
corr(z_a, z_b) for every role pair on the same team and across the two teams,
pooling all games. Players in different games are independent.

For simulation, a game's correlation matrix is built from these role pairs
(players beyond the primary roles borrow the nearest role) and projected to the
nearest valid correlation matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR

ROLES = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "K", "DST"]
CORR_PATH = DATA_DIR / "models" / "correlation.json"
# depth slots per position, in projection order
SLOTS = {"QB": ["QB1"], "RB": ["RB1", "RB2"], "WR": ["WR1", "WR2", "WR3"], "TE": ["TE1"], "K": ["K"], "D/ST": ["DST"]}


def game_key() -> pl.Expr:
    lo = pl.min_horizontal("team", "opp")
    hi = pl.max_horizontal("team", "opp")
    return pl.concat_str([pl.col("season"), pl.col("week"), lo, hi], separator="-")


def assign_roles(df: pl.DataFrame) -> pl.DataFrame:
    """role = primary slot by projection rank within team-week; extras get the last slot of their position."""
    df = df.with_columns(pl.col("proj").rank("ordinal", descending=True).over("season", "week", "team", "pos")
                         .cast(pl.Int64).alias("depth"))
    role = pl.lit(None, dtype=pl.String)
    for pos, slots in SLOTS.items():
        for i, s in enumerate(slots, start=1):
            role = pl.when((pl.col("pos") == pos) & (pl.col("depth") == i)).then(pl.lit(s)).otherwise(role)
        role = pl.when((pl.col("pos") == pos) & (pl.col("depth") > len(slots))).then(pl.lit(slots[-1])).otherwise(role)
    primary = pl.col("depth") <= pl.col("pos").replace_strict({p: len(s) for p, s in SLOTS.items()}, default=0)
    return df.with_columns(role.alias("role"), primary.alias("primary"), game_key().alias("game"))


def estimate(df: pl.DataFrame) -> dict:
    """df needs: game, team, role, primary, z. Returns {'same': 9x9, 'opp': 9x9, 'n_same', 'n_opp'}."""
    d = df.filter(pl.col("primary")).select("game", "team", "role", "z")
    wide = d.pivot(on="role", index=["game", "team"], values="z", aggregate_function="first")
    for r in ROLES:
        if r not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(r))
    # pair each team with its opponent in the same game
    opp = wide.rename({r: f"o_{r}" for r in ROLES}).rename({"team": "opp_team"})
    pairs = wide.join(opp, on="game").filter(pl.col("team") != pl.col("opp_team"))
    k = len(ROLES)
    same, opp_m = np.eye(k), np.zeros((k, k))
    n_same, n_opp = np.zeros((k, k), int), np.zeros((k, k), int)
    for i, a in enumerate(ROLES):
        for j, b in enumerate(ROLES):
            if i < j:
                x = wide.select(a, b).drop_nulls()
                if x.height > 30:
                    same[i, j] = same[j, i] = np.corrcoef(x[a], x[b])[0, 1]
                    n_same[i, j] = n_same[j, i] = x.height
            x = pairs.select(a, f"o_{b}").drop_nulls()
            if x.height > 30:
                opp_m[i, j] = np.corrcoef(x[a], x[f"o_{b}"])[0, 1]
                n_opp[i, j] = x.height
    opp_m = (opp_m + opp_m.T) / 2  # each game counted from both sides; symmetrize
    return {"roles": ROLES, "same": same.tolist(), "opp": opp_m.tolist(),
            "n_same": n_same.tolist(), "n_opp": n_opp.tolist()}


def save(c: dict, path: Path = CORR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=1))


def load(path: Path = CORR_PATH) -> dict:
    return json.loads(path.read_text())


# a duplicate of a role (e.g. two WR3s) correlates with it like the adjacent depth slot does
_NEIGHBOR = {"QB1": "QB1", "RB1": "RB2", "RB2": "RB1", "WR1": "WR2", "WR2": "WR3", "WR3": "WR2",
             "TE1": "TE1", "K": "K", "DST": "DST"}


def game_matrix(c: dict, roles: list[str], sides: list[int]) -> np.ndarray:
    """Correlation matrix for the players of one game. sides: 0/1 for the two teams."""
    idx = {r: i for i, r in enumerate(c["roles"])}
    same, opp = np.array(c["same"]), np.array(c["opp"])
    n = len(roles)
    m = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            ra, rb = roles[a], roles[b]
            if sides[a] == sides[b]:
                v = same[idx[ra], idx[_NEIGHBOR[rb]]] if ra == rb else same[idx[ra], idx[rb]]
                if ra == rb and ra in ("QB1", "TE1", "K", "DST"):
                    v = 0.0  # backup QB/TE etc.: treat as unrelated
            else:
                v = opp[idx[ra], idx[rb]]
            m[a, b] = m[b, a] = v
    return nearest_correlation(m)


def nearest_correlation(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues and rescale to unit diagonal."""
    w, v = np.linalg.eigh((m + m.T) / 2)
    if w.min() >= eps:
        return m
    m2 = (v * np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(m2))
    return m2 / d[:, None] / d[None, :]


def fit_live() -> dict:
    """Estimate on 2019-2025 using each player's range position under the live marginals."""
    from scipy.stats import norm

    from .marginals import MODEL_PATH, Marginals, cdf
    frame = pl.read_parquet(DATA_DIR / "marginal_frame.parquet")
    u = cdf(Marginals.load(MODEL_PATH).quantiles(frame), frame["actual"].to_numpy(), np.random.default_rng(0))
    frame = assign_roles(frame.with_columns(pl.Series("z", norm.ppf(np.clip(u, 1e-4, 1 - 1e-4)))))
    c = estimate(frame)
    save(c)
    return c
