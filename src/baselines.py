"""The three reference models. None of them is fitted to the training set.

Each one reads only the history of the sample it predicts, so the folds do not change
their predictions. Input: data.Histories. Output: (N, 52) in dB.
"""

import numpy as np

from src.data import Histories

EMA_WEIGHT = 0.5  # fixed a priori (PROTOCOL.md section 4); not tuned to the data


def persistence(h: Histories) -> np.ndarray:
    """The last exam is the prediction of the next one: assumes nothing changed."""
    return h.last.copy()


def plr(h: Histories) -> np.ndarray:
    """Pointwise linear regression: one least-squares line per location, extrapolated
    to the date of the target.

    Time is in years, with the last exam at t = 0, so the target is at t = delta_t.
    With 2 exams the line goes exactly through both.
    """
    pred = np.empty_like(h.target)
    for i, k in enumerate(h.n_hist):
        years = np.cumsum(h.gap_months[i, :k]) / 12
        t = years - years[-1]
        y = h.sens[i, :k]

        centered = t - t.mean()
        slope = centered @ (y - y.mean(axis=0)) / np.dot(centered, centered)
        intercept = y.mean(axis=0) - slope * t.mean()
        pred[i] = intercept + slope * (h.delta_t_months[i] / 12)
    return pred


def ema(h: Histories, weight: float = EMA_WEIGHT) -> np.ndarray:
    """Exponential moving average, one per location.

    estimate <- estimate + weight * (new exam - estimate)

    With weight 1/2 the last exam weighs 1/2, the one before 1/4, then 1/8, ...
    The prediction is the current estimate: no trend.
    """
    pred = np.empty_like(h.target)
    for i, k in enumerate(h.n_hist):
        level = h.sens[i, 0].copy()
        for exam in h.sens[i, 1:k]:
            level += weight * (exam - level)
        pred[i] = level
    return pred


BASELINES = {"persistence": persistence, "plr": plr, "ema": ema}
