"""Bootstrap by patient, and the progression groups on the real cohort."""

import numpy as np
import pytest

from src import config
from src.evaluate import PatientBootstrap, bias, progression_group, sample_error


def test_error_and_bias_signs():
    target = np.full((1, 52), 20.0)
    assert sample_error(target + 2, target)[0] == 2.0
    assert bias(target + 2, target)[0] == 2.0  # predicts better than measured: optimistic


def test_bootstrap_of_a_constant_has_no_width():
    est, low, high = PatientBootstrap(np.arange(50)).mean(np.full(50, 0.3))
    assert est == pytest.approx(0.3) and low == pytest.approx(0.3) and high == pytest.approx(0.3)


def test_bootstrap_resamples_patients_not_samples():
    # 2 patients, the second with 99 samples. Resampling samples would give a narrow
    # interval around 1; resampling patients sometimes draws only patient 0.
    pat_id = np.array([0] + [1] * 99)
    values = np.array([0.0] + [1.0] * 99)
    _, low, high = PatientBootstrap(pat_id).mean(values)
    assert low == 0.0 and high == 1.0


def test_ratio_uses_the_same_resample_for_both_terms():
    rng = np.random.default_rng(0)
    x = rng.normal(1, 1, 200)
    est, low, high = PatientBootstrap(np.arange(200)).ratio(2 * x, x)
    assert (est, low, high) == pytest.approx((2.0, 2.0, 2.0))


@pytest.mark.skipif(not config.SAMPLES_PARQUET.exists(), reason="run scripts/01_prepare_data.py first")
def test_rapid_group_size_matches_the_paper():
    import polars as pl

    last = pl.read_parquet(config.SAMPLES_PARQUET).sort("sample_id")["is_last"].to_numpy()
    group = progression_group()
    assert int((last & (group == "rapid")).sum()) == 41
