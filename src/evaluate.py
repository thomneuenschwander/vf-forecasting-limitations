"""Errors, confidence intervals and progression groups (PROTOCOL.md section 5).

All results use the evaluation set: the last sample of each series (`is_last`), so that
long series do not weigh more.

    error of a sample  = mean |prediction - measured| over the 52 locations (dB)
    MAE of a model     = mean error over the samples
    gain of a model    = MAE(persistence) - MAE(model); positive = better than persistence

95% confidence intervals: 2,000 bootstrap resamples of PATIENTS (the two eyes of a patient
are not independent). A difference between two models is significant when the interval
of the paired difference does not contain zero.
"""

import numpy as np
import polars as pl
from scipy import stats

from src import config

GROUPS = ("rapid", "moderate", "slow", "stable", "improving", "undetermined")
MIN_POINTS = 4  # exams needed to fit the MTD line


def sample_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,) mean absolute error over the 52 locations of each sample."""
    return np.abs(pred - target).mean(axis=1)


def bias(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N,) predicted minus measured mean sensitivity. Positive = optimistic."""
    return pred.mean(axis=1) - target.mean(axis=1)


class PatientBootstrap:
    """Resamples patients with replacement. Built once per set of samples.

    The mean over a resample is (sum of the drawn patients' totals) / (their sample counts),
    so each replicate is a sum over ~2,000 patients instead of a rebuild of the sample list.
    """

    def __init__(self, pat_id: np.ndarray, replicates: int = config.N_BOOTSTRAP, seed: int = config.SEED):
        _, self.codes = np.unique(pat_id, return_inverse=True)
        self.n = int(self.codes.max()) + 1
        self.counts = np.bincount(self.codes, minlength=self.n).astype(np.float64)
        self.draws = np.random.default_rng(seed).integers(0, self.n, size=(replicates, self.n))

    def _totals(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.codes, weights=values, minlength=self.n)

    def mean(self, values: np.ndarray) -> tuple[float, float, float]:
        """(estimate, low, high) of the mean."""
        replicas = self._totals(values)[self.draws].sum(axis=1) / self.counts[self.draws].sum(axis=1)
        return float(values.mean()), *map(float, np.percentile(replicas, [2.5, 97.5]))

    def ratio(self, top: np.ndarray, bottom: np.ndarray) -> tuple[float, float, float]:
        """(estimate, low, high) of mean(top) / mean(bottom), both from the same resample."""
        replicas = self._totals(top)[self.draws].sum(axis=1) / self._totals(bottom)[self.draws].sum(axis=1)
        return float(top.sum() / bottom.sum()), *map(float, np.percentile(replicas, [2.5, 97.5]))


def progression_group() -> np.ndarray:
    """Group of each sample (sample_id order) by how fast the field is getting worse.

    Slope, in dB/year, of a line fitted to the MTD (mean over the 52 locations of the total
    deviation: measured minus expected for age) against age. The fit uses the history
    WITHOUT its last exam, and never the target: a low value by chance in either exam
    would otherwise decide both the group and the error. At least 4 exams are needed, so
    only samples with 5+ exams of history get a group; the others are "undetermined".

        rapid   slope <= -1 dB/year and p < 0.05
        stable  p >= 0.05
    """
    exams = pl.read_parquet(config.EXAMS_PARQUET)
    td = np.asarray(exams["td"].to_list(), dtype=float)[:, list(config.TESTABLE_INDEX)]
    exams = exams.select("block_id", "order", "age").with_columns(mtd=pl.Series(td.mean(axis=1)))
    samples = pl.read_parquet(config.SAMPLES_PARQUET).sort("sample_id")

    points = (
        samples.select("sample_id", "block_id", "hist_start_order", "hist_end_order")
        .join(exams, on="block_id")
        .filter(pl.col("order").is_between("hist_start_order", "hist_end_order", closed="left"))
        .group_by("sample_id")
        .agg("age", "mtd")
    )
    groups = {}
    for sample_id, age, mtd in points.iter_rows():
        if len(age) < MIN_POINTS:
            continue
        fit = stats.linregress(age, mtd)
        if fit.pvalue >= config.PROGRESSION_ALPHA:
            groups[sample_id] = "stable"
        elif fit.slope > 0:
            groups[sample_id] = "improving"
        elif fit.slope <= config.SLOPE_RAPID:
            groups[sample_id] = "rapid"
        elif fit.slope <= config.SLOPE_MODERATE:
            groups[sample_id] = "moderate"
        else:
            groups[sample_id] = "slow"
    return np.array([groups.get(s, "undetermined") for s in samples["sample_id"]], dtype=object)
