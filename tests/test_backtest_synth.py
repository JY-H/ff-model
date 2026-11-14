import numpy as np
import polars as pl
import pytest

from ff_model.model.backtest_synth import week_quantiles


def test_week_quantiles_uses_that_weeks_curve():
    # the same player in two weeks with very different curves
    frame = pl.DataFrame({"player_id": [7, 7, 8], "week": [1, 2, 1]}).with_row_index("_qi")
    q = np.array([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]])
    w1 = week_quantiles(frame.filter(pl.col("week") == 1), q)
    w2 = week_quantiles(frame.filter(pl.col("week") == 2), q)
    assert w1[7].tolist() == [1.0, 2.0] and w2[7].tolist() == [10.0, 20.0]


def test_week_quantiles_rejects_duplicate_players():
    frame = pl.DataFrame({"player_id": [7, 7], "week": [1, 1]}).with_row_index("_qi")
    with pytest.raises(ValueError):
        week_quantiles(frame, np.zeros((2, 2)))
