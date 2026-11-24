"""Filters and samples on tiny hand-made series, plus the paper's counts on the real cohort."""

import polars as pl
import pytest

from src import config, data


def series(ages: list[float], pat_id: int = 1, eye: str = "R", sens: float = 30.0) -> pl.DataFrame:
    n = len(ages)
    return pl.DataFrame(
        {
            "pat_id": [pat_id] * n,
            "eye": [eye] * n,
            "order": list(range(1, n + 1)),
            "age": ages,
            "sens": [[sens] * config.N_LOCATIONS] * n,
        },
        schema_overrides={"sens": pl.Array(pl.Float64, config.N_LOCATIONS)},
    )


def months(*m: float) -> list[float]:
    return [50 + x / 12 for x in m]


def test_e1_removes_exam_with_all_tested_locations_at_zero():
    exams = pl.concat([series([50.0]), series([51.0], sens=0.0).with_columns(order=pl.lit(2, pl.Int64))])
    assert data.f1_all_zero(exams)["age"].to_list() == [50.0]


def test_e2_removes_series_with_baseline_age_below_one_year():
    exams = pl.concat([series([0.5, 1.5, 2.5], pat_id=1), series([40.0, 41.0, 42.0], pat_id=2)])
    assert data.f2_baseline_age(exams)["pat_id"].unique().to_list() == [2]


def test_e3_compares_with_the_last_kept_exam_not_the_previous_one():
    # 0, 2, 4 months: 2 is dropped (< 3 after 0); 4 is kept (>= 3 after 0, although 2 after 2).
    kept = data.f3_min_gap(series(months(0, 2, 4)))
    assert kept["order"].to_list() == [1, 3]


def test_e4_splits_on_gaps_longer_than_three_years():
    split = data.f4_split_long_gaps(series(months(0, 12, 60, 72)))
    assert split["block_id"].to_list() == ["1_R_1", "1_R_1", "1_R_2", "1_R_2"]


def test_e5_drops_series_shorter_than_three_exams():
    exams = data.f4_split_long_gaps(series(months(0, 12, 60, 72, 84)))
    assert data.f5_short_series(exams)["block_id"].unique().to_list() == ["1_R_2"]


def test_expanding_window_samples():
    exams = data.f4_split_long_gaps(series(months(0, 6, 12, 24)))
    samples = data.build_samples(exams)
    # 4 exams, 2 of minimum history -> targets are exams 3 and 4
    assert samples["target_order"].to_list() == [3, 4]
    assert samples["n_hist"].to_list() == [2, 3]
    assert samples["is_last"].to_list() == [False, True]
    assert samples["delta_t_months"].round(6).to_list() == [6.0, 12.0]


# ------------------------------------------------------------- real cohort (after step 1)

needs_cohort = pytest.mark.skipif(
    not config.SAMPLES_PARQUET.exists(), reason="run scripts/01_prepare_data.py first"
)


@needs_cohort
def test_cohort_matches_the_paper():
    exams = pl.read_parquet(config.EXAMS_PARQUET)
    samples = pl.read_parquet(config.SAMPLES_PARQUET)
    assert exams.height == 20_052
    assert exams["block_id"].n_unique() == 4_378
    assert exams.select("pat_id", "eye").n_unique() == 3_980
    assert exams["pat_id"].n_unique() == 2_093
    assert samples.height == 11_296
    assert samples["is_last"].sum() == 4_378


@needs_cohort
def test_each_patient_is_in_exactly_one_fold():
    exams = pl.read_parquet(config.EXAMS_PARQUET)
    assert exams.group_by("pat_id").agg(pl.col("fold").n_unique())["fold"].max() == 1
    assert sorted(exams["fold"].unique().to_list()) == [1, 2, 3, 4, 5]
