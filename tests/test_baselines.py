"""The three baselines on hand-made histories whose answer is known."""

import numpy as np
import pytest

from src import baselines, config
from src.data import Histories


def histories(values: list[float], gaps: list[float], delta_t: float = 12.0) -> Histories:
    """One sample whose 52 locations all follow `values`."""
    k = len(values)
    sens = np.repeat(np.asarray(values, dtype=float)[None, :, None], config.N_TESTABLE, axis=2)
    return Histories(
        sample_id=np.array([0]),
        pat_id=np.array([1]),
        fold=np.array([1]),
        is_last=np.array([True]),
        n_hist=np.array([k]),
        sens=sens,
        gap_months=np.asarray([gaps]),
        delta_t_months=np.array([delta_t]),
        age_last=np.array([60.0]),
        target=np.zeros((1, config.N_TESTABLE)),
    )


def test_persistence_repeats_the_last_exam():
    h = histories([30, 28, 25], [0, 12, 12])
    assert np.all(baselines.persistence(h) == 25)


def test_plr_extrapolates_the_line_to_the_target_date():
    # -2 dB/year, exams 6 months apart, target 1 year after the last: 26 - 2 = 24
    h = histories([28, 27, 26], [0, 6, 6], delta_t=12)
    assert baselines.plr(h) == pytest.approx(np.full((1, config.N_TESTABLE), 24.0))


def test_plr_with_two_exams_goes_through_both():
    h = histories([30, 27], [0, 12], delta_t=6)
    assert baselines.plr(h) == pytest.approx(np.full((1, config.N_TESTABLE), 25.5))


def test_ema_weights_one_half_one_quarter():
    # 20, then 40: (20 + 40) / 2 = 30; then 10: (30 + 10) / 2 = 20
    h = histories([20, 40, 10], [0, 12, 12])
    assert np.all(baselines.ema(h) == 20)


def test_ema_ignores_the_padding():
    h = histories([20, 40, 10], [0, 12, 12])
    h.n_hist = np.array([2])  # only 20 and 40 are history
    assert np.all(baselines.ema(h) == 30)
