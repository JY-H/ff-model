import numpy as np
from scipy.stats import norm

from ff_model.model import correlation as C
from ff_model.model.marginals import TAUS
from ff_model.model.simulate import draw

R = C.ROLES
CORR = {"roles": R, "same": np.eye(len(R)).tolist(), "opp": np.zeros((len(R), len(R))).tolist()}
qb_wr1 = (R.index("QB1"), R.index("WR1"))
CORR["same"][qb_wr1[0]][qb_wr1[1]] = CORR["same"][qb_wr1[1]][qb_wr1[0]] = 0.4
d_qb = (R.index("DST"), R.index("QB1"))
CORR["opp"][d_qb[0]][d_qb[1]] = CORR["opp"][d_qb[1]][d_qb[0]] = -0.5


def _q(n):  # standard-normal quantile curves
    return np.tile(norm.ppf(TAUS), (n, 1))


def test_marginals_preserved_and_correlation_applied():
    roles, sides = ["QB1", "WR1", "DST"], [0, 0, 1]
    x = draw(_q(3), [0, 0, 0], roles, sides, CORR, n=40000, rng=np.random.default_rng(0))
    assert np.allclose(np.median(x, axis=0), 0, atol=0.03)
    assert np.allclose(np.quantile(x, 0.9, axis=0), norm.ppf(0.9), atol=0.05)
    c = np.corrcoef(x.T)
    assert abs(c[0, 1] - 0.4) < 0.03    # QB-WR1 same team
    assert abs(c[0, 2] + 0.5) < 0.03    # opposing D/ST vs QB


def test_different_games_are_independent_and_duplicates_are_valid():
    roles, sides = ["QB1", "WR1", "WR1", "QB1"], [0, 0, 0, 0]
    x = draw(_q(4), [1, 1, 1, 2], roles, sides, CORR, n=20000, rng=np.random.default_rng(1))
    c = np.corrcoef(x.T)
    assert abs(c[0, 3]) < 0.03          # QB in another game
    m = C.game_matrix(CORR, ["WR1", "WR1", "QB1", "QB1"], [0, 0, 0, 0])
    assert np.all(np.linalg.eigvalsh(m) > 0) and np.allclose(np.diag(m), 1)
