"""The networks: size of the paper's configurations, and padding never leaks into a prediction."""

import numpy as np
import pytest
import torch

from src import config, nets
from src.data import Histories

needs_grid = pytest.mark.skipif(not config.GRID_PARQUET.exists(), reason="run scripts/01_prepare_data.py first")


@needs_grid
@pytest.mark.parametrize("name, expected", [("lstm", 199_836), ("convlstm", 37_261)])
def test_parameter_counts_match_table_1(name, expected):
    assert nets.n_parameters(name, nets.load_hparams(name)) == expected


def random_histories(n_hist: list[int], seed: int = 0) -> Histories:
    rng = np.random.default_rng(seed)
    n, steps = len(n_hist), max(n_hist)
    sens = np.zeros((n, steps, config.N_TESTABLE))
    gaps = np.zeros((n, steps))
    for i, k in enumerate(n_hist):
        sens[i, :k] = rng.uniform(0, 35, (k, config.N_TESTABLE))
        gaps[i, 1:k] = rng.uniform(3, 24, k - 1)
    return Histories(
        sample_id=np.arange(n), pat_id=np.arange(n), fold=np.ones(n, int), is_last=np.ones(n, bool),
        n_hist=np.asarray(n_hist), sens=sens, gap_months=gaps,
        delta_t_months=rng.uniform(3, 24, n), age_last=rng.uniform(40, 80, n),
        target=rng.uniform(0, 35, (n, config.N_TESTABLE)),
    )


@needs_grid
@pytest.mark.parametrize("name", ["lstm", "convlstm"])
def test_prediction_does_not_depend_on_the_rest_of_the_batch(name):
    """A sample with 2 exams predicted alone, or next to one with 9, gives the same output."""
    torch.manual_seed(0)
    net = nets.build(name, {"hidden": 8, "layers": 2, "dropout": 0.0}).eval()
    t = nets.to_tensors(random_histories([2, 9, 4]), torch.device("cpu"))
    together = nets.predict(net, t, np.array([0, 1, 2]))
    alone = nets.predict(net, t, np.array([0]))
    assert np.allclose(together[0], alone[0], atol=1e-5)


def test_max_history_keeps_only_the_last_exams():
    h = random_histories([4])
    t = nets.to_tensors(h, torch.device("cpu"), max_history=1)
    assert t.lengths_host.tolist() == [1]
    assert np.allclose(nets.denormalize(t.seq[0, 0, :52].numpy()), h.sens[0, 3], atol=1e-4)
    assert t.seq[0, 0, 52].item() == 0.0  # no previous exam visible
