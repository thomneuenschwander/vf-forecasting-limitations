"""Step 2: predictions of the three reference models.

    uv run python scripts/02_run_baselines.py

Writes outputs/preds/{persistence,plr,ema}.parquet. No training, no GPU: each
model reads only the history of the sample it predicts.
"""

import numpy as np

from src import data
from src.baselines import BASELINES


def main() -> None:
    h = data.load_histories()
    for name, model in BASELINES.items():
        data.write_predictions(name, h, model(h))
        pred = data.read_predictions(name)  # as saved, after clipping to 0-50 dB
        mae = np.abs(pred - h.target)[h.is_last].mean()
        print(f"{name:12s} MAE on the evaluation set: {mae:.3f} dB")


if __name__ == "__main__":
    main()
