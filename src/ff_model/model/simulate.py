"""Monte Carlo draws of player points: per-player quantile curves + game-level copula."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .correlation import game_matrix
from .marginals import TAUS

_T = np.concatenate([[0.0], TAUS, [1.0]])


def _extend(q: np.ndarray) -> np.ndarray:
    """Add linear tails at tau = 0 and 1, matching marginals.cdf."""
    lo_w = np.maximum(q[:, 1] - q[:, 0], 1e-6) * TAUS[0] / (TAUS[1] - TAUS[0])
    hi_w = np.maximum(q[:, -1] - q[:, -2], 1e-6) * (1 - TAUS[-1]) / (TAUS[-1] - TAUS[-2])
    return np.column_stack([q[:, 0] - lo_w, q, q[:, -1] + hi_w])


def inverse(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Points for uniforms u (n_draws x n_players) from each player's quantile curve q (n_players x taus)."""
    full = _extend(q)
    out = np.empty_like(u)
    for j in range(q.shape[0]):
        out[:, j] = np.interp(u[:, j], _T, full[j])
    return out


def draw(q: np.ndarray, games: list, roles: list[str], sides: list[int], corr: dict | None,
         n: int = 20000, rng: np.random.Generator | None = None) -> np.ndarray:
    """(n, n_players) simulated points. corr=None draws players independently."""
    rng = rng or np.random.default_rng()
    p = q.shape[0]
    z = rng.standard_normal((n, p))
    if corr is not None:
        games = np.asarray(games)
        for g in np.unique(games):
            idx = np.flatnonzero(games == g)
            if len(idx) < 2:
                continue
            m = game_matrix(corr, [roles[i] for i in idx], [sides[i] for i in idx])
            z[:, idx] = z[:, idx] @ np.linalg.cholesky(m).T
    return inverse(q, norm.cdf(z))
