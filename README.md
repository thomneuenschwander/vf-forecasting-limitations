# Limitations of deep learning for visual field forecasting

Code for the paper *Limitações do Aprendizado Profundo na Previsão de Campos Visuais*.

Question: when a network predicts the next visual field better than repeating the last
exam, is it anticipating progression or just smoothing out noise? We compare an LSTM and
a ConvLSTM with three references that learn nothing from the data: persistence (repeat
the last exam), pointwise linear regression (pLR) and an exponential moving average (EMA).

Data: public UWHVF dataset, 4,378 series. The full protocol is in [PROTOCOL.md](PROTOCOL.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (tested on a GTX 1660, 6 GB).

```bash
uv sync
```

Download the data into `data/raw/` (see [data/README.md](data/README.md)), then:

| Step | Command | Time |
|---|---|---|
| 1. Cohort, samples, folds | `uv run python scripts/01_prepare_data.py` | 10 s |
| 2. Reference models | `uv run python scripts/02_run_baselines.py` | 5 s |
| 3. Hyperparameter search (optional) | `uv run python scripts/03_tune.py --model convlstm` | ~5 h |
| 4. Networks | `uv run python scripts/04_train_nets.py` | ~35 min |
| 4b. ConvLSTM, last exam only | `uv run python scripts/04_train_nets.py --model convlstm --max-history 1` | ~10 min |
| 5. Results | `uv run python scripts/05_evaluate.py` | 15 s |

Step 3 can be skipped: `configs/` already holds the hyperparameters used in the paper.
Step 5 writes the tables to `results/`. Tests: `uv run pytest`.

## Expected results

Evaluation set: last exam of each series (n = 4,378). Gain = MAE of persistence minus
MAE of the model. Intervals: 95%, bootstrap by patient.

| Model | MAE (dB) | Gain (dB) |
|---|---|---|
| Persistence | 2.99 | — |
| pLR | 4.15 | −1.15 [−1.23; −1.08] |
| EMA | 2.70 | 0.30 [0.26; 0.33] |
| LSTM | 2.67 | 0.32 [0.30; 0.35] |
| ConvLSTM | 2.52 | 0.47 [0.44; 0.50] |
| ConvLSTM, last exam only | 2.68 | 0.32 [0.30; 0.34] |

Steps 1, 2 and 5 are deterministic. Training on the GPU is not: a new run of step 4
can change the MAE of a network by about 0.01 dB.

## Layout

```
src/        config.py (every fixed choice), data.py, baselines.py, nets.py, evaluate.py
scripts/    the 5 steps above, run in order
configs/    hyperparameters of the LSTM and the ConvLSTM
results/    tables produced by step 5 (versioned)
outputs/    intermediate files: cohort, predictions, logs (not versioned)
tests/      checks against the numbers of the paper
```
