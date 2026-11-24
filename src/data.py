"""From the UWHVF files to the study cohort, the samples and the folds.

    load_exams      alldata.json -> one row per exam
    apply_filters   F1..F5 (PROTOCOL.md section 2)
    build_samples   expanding window, one-step-ahead (section 3)
    assign_folds    5 folds by patient (section 3)
    load_histories  the samples as arrays, ready for the models

exams.parquet keeps the 54 locations of the file. Every model sees and predicts only the
52 tested ones (config.TESTABLE_INDEX); the two blind spots are dropped in load_histories.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass

import numpy as np
import polars as pl

from src import config

SERIES = ("pat_id", "eye")


# --------------------------------------------------------------------------- loading


def load_exams() -> pl.DataFrame:
    """One row per exam: pat_id, eye, order (1, 2, ... by age), age, gender, sens, td."""
    raw = json.loads(config.ALLDATA_JSON.read_text(encoding="utf-8"))

    rows = []
    for pat_id, patient in raw["data"].items():
        for eye in ("R", "L"):
            # The JSON does not guarantee chronological order: sort by age.
            exams = sorted(patient.get(eye, []), key=lambda exam: exam["age"])
            for order, exam in enumerate(exams, start=1):
                td = list(exam["td_seq"])
                for i in config.BLIND_SPOT_INDEX:
                    td[i] = None  # the JSON repeats the sensitivity here; TD is undefined
                rows.append(
                    {
                        "pat_id": int(pat_id),
                        "eye": eye,
                        "order": order,
                        "age": exam["age"],
                        "gender": patient.get("gender"),
                        "sens": exam["hvf_seq"],
                        "td": td,
                    }
                )

    schema = {
        "pat_id": pl.Int32,
        "eye": pl.String,
        "order": pl.Int16,
        "age": pl.Float64,
        "gender": pl.String,
        "sens": pl.Array(pl.Float64, config.N_LOCATIONS),
        "td": pl.Array(pl.Float64, config.N_LOCATIONS),
    }
    exams = pl.DataFrame(rows, schema=schema).sort("pat_id", "eye", "order")

    n_eyes = exams.select(pl.struct(*SERIES)).n_unique()
    expected = (raw["pts"], raw["eyes"], raw["hvfs"])
    found = (exams["pat_id"].n_unique(), n_eyes, exams.height)
    assert found == expected, f"counts differ from the file header: {found} != {expected}"
    gaps = exams.select(pl.col("age").diff().over(*SERIES)).drop_nulls()
    assert (gaps["age"] > 0).all(), "ages not strictly increasing within an eye"
    return exams


def load_grid() -> pl.DataFrame:
    """Geometry of the 54 locations: row/column on the 8x9 grid, degrees, cluster."""
    raw = json.loads(config.ALLDATA_JSON.read_text(encoding="utf-8"))
    grid_pos = {c["index"]: (c["x"], c["y"]) for c in raw["coords"]}
    with config.COORD_CSV.open(newline="", encoding="utf-8") as fh:
        meta = {int(r["LocID"]) - 1: r for r in csv.DictReader(fh)}

    grid = pl.DataFrame(
        [
            {
                "index": i,
                "grid_row": grid_pos[i][0],
                "grid_col": grid_pos[i][1],
                "x_deg": int(meta[i]["X"]),
                "y_deg": int(meta[i]["Y"]),
                "cluster": int(meta[i]["Cluster"]),
            }
            for i in range(config.N_LOCATIONS)
        ]
    )
    blind = tuple(grid.filter(pl.col("cluster") == 0)["index"])
    assert blind == config.BLIND_SPOT_INDEX, f"blind spots at {blind}"
    return grid


# --------------------------------------------------------------------------- filters


def f1_all_zero(exams: pl.DataFrame) -> pl.DataFrame:
    """F1: remove exams with the 52 tested locations at 0 dB."""
    all_zero = pl.all_horizontal(pl.col("sens").arr.get(i) == 0.0 for i in config.TESTABLE_INDEX)
    return exams.filter(~all_zero)


def f2_baseline_age(exams: pl.DataFrame) -> pl.DataFrame:
    """F2: remove series whose first exam is at age < 1 year (recording error)."""
    return exams.filter(pl.col("age").min().over(*SERIES) >= config.MIN_BASELINE_AGE_YEARS)


def f3_min_gap(exams: pl.DataFrame) -> pl.DataFrame:
    """F3: remove exams taken < 3 months after the previous KEPT exam (repeat tests).

    Sequential: whether an exam stays depends on which earlier exams stayed.
    """
    exams = exams.sort(*SERIES, "order")
    keep, current, last_age = [], None, 0.0
    for pat_id, eye, age in exams.select("pat_id", "eye", "age").iter_rows():
        if (pat_id, eye) != current:
            keep.append(True)
            current, last_age = (pat_id, eye), age
        elif (age - last_age) * 12 >= config.MIN_GAP_MONTHS:
            keep.append(True)
            last_age = age
        else:
            keep.append(False)
    return exams.filter(pl.Series(keep))


def f4_split_long_gaps(exams: pl.DataFrame) -> pl.DataFrame:
    """F4: split a series wherever there is a gap > 3 years. Creates `block_id`.

    From here on a "series" is a block: pat_eye_k, the k-th piece of that eye.
    """
    gap_months = (pl.col("age") - pl.col("age").shift(1).over(*SERIES)) * 12
    piece = (gap_months > config.MAX_GAP_MONTHS).fill_null(False).cum_sum().over(*SERIES) + 1
    return exams.sort(*SERIES, "order").with_columns(
        block_id=pl.concat_str(
            pl.col("pat_id").cast(pl.String), pl.col("eye"), piece.cast(pl.String), separator="_"
        )
    )


def f5_short_series(exams: pl.DataFrame) -> pl.DataFrame:
    """F5: remove series with fewer than 3 exams."""
    return exams.filter(pl.len().over("block_id") >= config.MIN_EXAMS_PER_SERIES)


FILTERS = (
    ("F1", "all 52 tested locations at 0 dB", f1_all_zero),
    ("F2", "baseline age < 1 year", f2_baseline_age),
    ("F3", "exam < 3 months after the previous one", f3_min_gap),
    ("F4", "gap > 3 years splits the series", f4_split_long_gaps),
    ("F5", "series with < 3 exams", f5_short_series),
)


def _counts(exams: pl.DataFrame) -> dict[str, int]:
    series_key = "block_id" if "block_id" in exams.columns else pl.struct(*SERIES)
    return {
        "exams": exams.height,
        "series": exams.select(series_key).n_unique(),
        "eyes": exams.select(pl.struct(*SERIES)).n_unique(),
        "patients": exams["pat_id"].n_unique(),
    }


def apply_filters(exams: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply F1..F5 in order. Returns the cohort and a log of what each filter did."""
    log = [{"step": "raw", "description": "UWHVF", **_counts(exams)}]
    for step, description, rule in FILTERS:
        exams = rule(exams)
        log.append({"step": step, "description": description, **_counts(exams)})

    gaps = exams.select(pl.col("age").diff().over("block_id") * 12).drop_nulls()["age"]
    assert gaps.min() >= config.MIN_GAP_MONTHS and gaps.max() <= config.MAX_GAP_MONTHS
    return exams.sort("block_id", "order"), pl.DataFrame(log)


# --------------------------------------------------------------------------- samples


def build_samples(exams: pl.DataFrame) -> pl.DataFrame:
    """Expanding window: history = exams 1..p-1 of the series, target = exam p.

    Only an index: the sensitivities stay in exams.parquet. `is_last` marks the last
    exam of each series, the only sample used in the evaluation.
    """
    exams = exams.sort("block_id", "order").with_columns(
        position=pl.int_range(1, pl.len() + 1).over("block_id"),
        length=pl.len().over("block_id"),
        hist_start_order=pl.col("order").first().over("block_id"),
        hist_end_order=pl.col("order").shift(1).over("block_id"),
        hist_end_age=pl.col("age").shift(1).over("block_id"),
    )
    samples = (
        exams.filter(pl.col("position") > config.MIN_HISTORY)
        .select(
            "block_id",
            "pat_id",
            "eye",
            pl.col("hist_start_order").cast(pl.Int16),
            pl.col("hist_end_order").cast(pl.Int16),
            pl.col("order").alias("target_order"),
            (pl.col("position") - 1).cast(pl.Int16).alias("n_hist"),
            ((pl.col("age") - pl.col("hist_end_age")) * 12).alias("delta_t_months"),
            (pl.col("position") == pl.col("length")).alias("is_last"),
        )
        .sort("block_id", "target_order")
        .with_row_index("sample_id")
    )

    n_series = exams["block_id"].n_unique()
    assert samples.height == exams.height - config.MIN_HISTORY * n_series
    assert samples["is_last"].sum() == n_series
    return samples


# --------------------------------------------------------------------------- folds


def _progression_group(exams: pl.DataFrame) -> pl.DataFrame:
    """Progression group of each eye, from the MTD slope that UWHVF ships.

    Used ONLY to balance the folds. It uses the whole original series, target included,
    so it is never used as a model input nor to report results. Eyes without a slope
    (fewer than 4 exams) are "undetermined".
    """
    summary = pl.read_csv(config.SERIES_SUMMARY_CSV, null_values=["NA"]).select(
        pat_id=pl.col("PatID").cast(pl.Int32),
        eye=pl.col("Eye").str.slice(0, 1),
        slope=pl.col("Slope_MTD"),
        pval=pl.col("pvalue_MTD"),
    )
    s, p = pl.col("slope"), pl.col("pval")
    group = (
        pl.when(s.is_null() | p.is_null()).then(pl.lit("undetermined"))
        .when(p >= config.PROGRESSION_ALPHA).then(pl.lit("stable"))
        .when(s > 0).then(pl.lit("improving"))
        .when(s <= config.SLOPE_RAPID).then(pl.lit("rapid"))
        .when(s <= config.SLOPE_MODERATE).then(pl.lit("moderate"))
        .otherwise(pl.lit("slow"))
    )
    return (
        exams.select("pat_id", "eye").unique()
        .join(summary, on=["pat_id", "eye"], how="left")
        .select("pat_id", "eye", group=group)
    )


# A patient takes the most severe group among their eyes.
SEVERITY = {"undetermined": -1, "stable": 0, "improving": 1, "slow": 2, "moderate": 3, "rapid": 4}
# Order in which the (group x length) cells are laid out before dealing patients to folds.
# It is the order used to produce the paper's folds; changing it changes the folds.
CELL_ORDER = {"stable": 0, "undetermined": 1, "slow": 2, "improving": 3, "moderate": 4, "rapid": 5}


def assign_folds(exams: pl.DataFrame) -> pl.DataFrame:
    """5 folds by patient, so both eyes of a patient always fall in the same fold.

    Stratified by (most severe progression group x length of the longest series): patients
    are grouped into cells, shuffled with a fixed seed, and dealt round-robin to folds 1..5.
    Each fold gets a comparable share of the rare rapid progressors.
    """
    severity_to_group = {v: k for k, v in SEVERITY.items()}

    patients = (
        _progression_group(exams)
        .group_by("pat_id")
        .agg(severity=pl.col("group").replace_strict(SEVERITY).max())
        .join(
            exams.group_by("block_id").agg(pl.col("pat_id").first(), n=pl.len())
            .group_by("pat_id").agg(longest=pl.col("n").max()),
            on="pat_id",
        )
        .with_columns(
            group=pl.col("severity").replace_strict(severity_to_group, return_dtype=pl.String),
            length_bucket=pl.when(pl.col("longest") >= 6).then(2)
            .when(pl.col("longest") >= 4).then(1)
            .otherwise(0),
        )
    )

    # Python's Mersenne Twister is stable across versions and platforms.
    ids = sorted(patients["pat_id"].to_list())
    random.Random(config.SEED).shuffle(ids)
    rank = {pat_id: r for r, pat_id in enumerate(ids)}

    folds = (
        patients.with_columns(
            cell=pl.col("group").replace_strict(CELL_ORDER),
            rank=pl.col("pat_id").replace_strict(rank),
        )
        .sort("cell", "length_bucket", "rank")
        .with_row_index("position")
        .select("pat_id", fold=(pl.col("position") % config.N_FOLDS + 1).cast(pl.Int8))
        .sort("pat_id")
    )
    assert folds.height == exams["pat_id"].n_unique()
    return folds


# --------------------------------------------------------------------------- arrays


@dataclass
class Histories:
    """All samples as arrays, in sample_id order. Sensitivities in dB, 52 tested locations.

    Histories are right-padded with zeros: sample i uses sens[i, :n_hist[i]].
    """

    sample_id: np.ndarray  # (N,)
    pat_id: np.ndarray  # (N,)
    fold: np.ndarray  # (N,)
    is_last: np.ndarray  # (N,) the sample used in the evaluation
    n_hist: np.ndarray  # (N,) exams in the history
    sens: np.ndarray  # (N, T, 52) history
    gap_months: np.ndarray  # (N, T) months since the previous exam; 0 for the first
    delta_t_months: np.ndarray  # (N,) months from the last exam to the target
    age_last: np.ndarray  # (N,) age at the last exam of the history, in years
    target: np.ndarray  # (N, 52) the exam to predict

    @property
    def last(self) -> np.ndarray:
        """(N, 52) the last exam of each history."""
        return self.sens[np.arange(len(self.n_hist)), self.n_hist - 1]


def load_histories() -> Histories:
    exams = pl.read_parquet(config.EXAMS_PARQUET).sort("block_id", "order")
    samples = pl.read_parquet(config.SAMPLES_PARQUET).sort("sample_id")

    testable = list(config.TESTABLE_INDEX)
    blocks = {}
    for (block_id,), frame in exams.group_by("block_id", maintain_order=True):
        blocks[block_id] = (
            frame["order"].to_numpy(),
            frame["age"].to_numpy(),
            np.asarray(frame["sens"].to_list(), dtype=np.float64)[:, testable],
        )

    n, steps = samples.height, int(samples["n_hist"].max())
    sens = np.zeros((n, steps, config.N_TESTABLE))
    gap = np.zeros((n, steps))
    age_last = np.zeros(n)
    target = np.zeros((n, config.N_TESTABLE))

    rows = samples.select("block_id", "hist_start_order", "hist_end_order", "target_order", "n_hist")
    for i, (block_id, start, end, target_order, k) in enumerate(rows.iter_rows()):
        order, age, values = blocks[block_id]
        span = (order >= start) & (order <= end)
        assert span.sum() == k and order[span].max() < target_order  # no future in the history
        sens[i, :k] = values[span]
        gap[i, 1:k] = np.diff(age[span]) * 12
        age_last[i] = age[span][-1]
        target[i] = values[order == target_order][0]

    return Histories(
        sample_id=samples["sample_id"].to_numpy(),
        pat_id=samples["pat_id"].to_numpy(),
        fold=samples["fold"].to_numpy(),
        is_last=samples["is_last"].to_numpy(),
        n_hist=samples["n_hist"].to_numpy().astype(np.int64),
        sens=sens,
        gap_months=gap,
        delta_t_months=samples["delta_t_months"].to_numpy(),
        age_last=age_last,
        target=target,
    )


def write_predictions(name: str, h: Histories, pred: np.ndarray) -> None:
    """outputs/preds/{name}.parquet: one row per sample, the 52 predicted values in dB.

    Every model's prediction is clipped to the range observed in UWHVF, 0-50 dB. It only
    matters for the pLR, whose lines can extrapolate far outside it.
    """
    assert pred.shape == h.target.shape and np.isfinite(pred).all()
    pred = np.clip(pred, config.SENS_MIN, config.SENS_MAX)
    config.PREDS_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"sample_id": h.sample_id, "fold": h.fold, "pred": pred},
        schema_overrides={"pred": pl.Array(pl.Float64, config.N_TESTABLE)},
    ).write_parquet(config.PREDS_DIR / f"{name}.parquet")


def read_predictions(name: str) -> np.ndarray:
    """(N, 52) in sample_id order."""
    preds = pl.read_parquet(config.PREDS_DIR / f"{name}.parquet").sort("sample_id")
    return np.asarray(preds["pred"].to_list())
