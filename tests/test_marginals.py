import numpy as np

from ff_model.model.marginals import TAUS, cdf


def test_cdf_continuous_and_point_mass():
    q = np.tile(np.linspace(0, 22, len(TAUS)), (3, 1))
    q[2, :5] = 0.0  # point mass at 0 covering the lowest 5 levels
    u = cdf(q, np.array([q[0, 11], 30.0, 0.0]), np.random.default_rng(1))
    assert abs(u[0] - 0.5) < 1e-6           # median maps to 0.5
    assert u[1] == 1.0                      # far above the top tail
    assert 0.0 <= u[2] <= TAUS[4] + 1e-6    # inside the mass at 0


def test_pit_is_uniform_when_model_is_right():
    rng = np.random.default_rng(0)
    from scipy.stats import norm
    q = np.tile(norm.ppf(TAUS), (20000, 1))
    u = cdf(q, rng.standard_normal(20000), rng)
    hist = np.histogram(u, bins=10, range=(0, 1))[0] / 20000
    assert np.all(np.abs(hist - 0.1) < 0.012)


def test_live_quantiles_rise_with_projection():
    import polars as pl
    import pytest
    from ff_model.model.marginals import MODEL_PATH, Marginals
    if not MODEL_PATH.exists():
        pytest.skip("no fitted model")
    m = Marginals.load(MODEL_PATH)
    for pos, lo, hi in [("K", 4, 13), ("D/ST", 1, 14), ("QB", 8, 32), ("RB", 3, 28), ("WR", 3, 28), ("TE", 2, 22)]:
        ps = np.linspace(lo, hi, 12)
        n = len(ps)
        ex = pl.DataFrame({"pos": [pos] * n, "proj": ps, "p_rec_td": ps * 0.03, "p_rush_td": ps * 0.0,
                           "p_rec": ps * (0.35 if pos in ("RB", "WR", "TE") else 0.0),
                           "p_pass_td": ps * (0.08 if pos == "QB" else 0.0), "p_rush_yds": ps * 0.0})
        q = m.quantiles(ex)
        assert np.all(np.diff(q, axis=0) >= -0.05), pos
