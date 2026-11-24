# Protocol

Every fixed value below is in `src/config.py`. The section numbers are the ones cited in
the code.

## 1. Data

UWHVF (Montesano et al., 2022): 28,943 exams of 24-2 visual fields from 3,871 patients.
Each exam has 54 locations. The two blind spots are dropped, so every model sees and
predicts **52 sensitivities** (dB). Step 1 checks the sha256 of the files.

## 2. Cohort

Five filters, applied in this order:

| Filter | Rule | Exams | Series | Patients |
|---|---|---|---|---|
| — | raw data | 28,943 | 7,428 | 3,871 |
| F1 | remove exams with the 52 locations at 0 dB | 28,932 | 7,426 | 3,871 |
| F2 | remove series whose first exam is at age < 1 year | 28,912 | 7,421 | 3,868 |
| F3 | remove exams < 3 months after the previous kept exam | 27,793 | 7,421 | 3,868 |
| F4 | split a series at every gap > 3 years | 27,793 | 9,056 | 3,868 |
| F5 | remove series with < 3 exams | **20,052** | **4,378** | **2,093** |

The cohort has 4,378 series from 3,980 eyes. Gaps between exams, and therefore the
prediction horizon, are between 3 months and 3 years.

## 3. Samples and cross-validation

**Samples.** Expanding window: in each series, exams 1..p−1 predict exam p, starting
with 2 exams to predict the 3rd. This gives 11,296 samples. All of them are used for
training. Only the **last sample of each series** (4,378) is used for evaluation, so
that long series do not weigh more.

**Folds.** 5 folds by patient: both eyes of a patient are always in the same fold. The
folds are stratified so that each one gets a similar share of each kind of patient:

- progression group of the patient's worst eye, from the MTD slope shipped with UWHVF
  (`Series_Summary.csv`);
- length of the patient's longest series: < 4, 4–5 or ≥ 6 exams.

Patients are shuffled within each cell with seed 20260101 and dealt to folds 1..5 in
turn. That slope uses the whole series, so it is used only to build the folds, never
as an input or in the results.

Each fold is predicted by models trained on the other 4. The union of the 5 test folds
gives one prediction per sample, always from a model that never saw that patient.

## 4. Models

All predictions are clipped to 0–50 dB, the range observed in UWHVF.

**References.** None of them is fitted to the data.

- **Persistence**: the next exam equals the last one.
- **pLR**: one least-squares line per location over the history, extrapolated to the
  date of the target.
- **EMA**: exponential moving average per location, `estimate ← estimate + ½ (exam −
  estimate)`. The last exam weighs ½, the previous one ¼, and so on. The weight ½ was
  fixed before looking at the results, not searched.

**Networks.** LSTM and ConvLSTM. Both predict the correction to the last exam:
`prediction = last exam + network(history)`.

- Input per exam: the 52 sensitivities and the interval since the previous exam. The
  LSTM reads them as a vector. The ConvLSTM places them on the 8×9 grid of the field
  (channels: sensitivity, mask of the 52 valid cells, interval) and uses 3×3
  convolutions, so it knows which locations are neighbors.
- After the last exam: the interval to the target and the age at the last exam.
- Scale: `(dB − 25) / 8`, fixed. It is not estimated from the data.

**Training**, identical for both networks: AdamW, L1 loss (the MAE itself), batches of
64, up to 600 epochs. Early stopping with patience 20 on a validation split of 20% of
the training patients.

**Hyperparameter search** (step 3, Optuna TPE). It uses only the training patients of
fold 1, never its test patients. Both networks get the same budget: 40 trials, no
pruning, 1,500 to 200,000 parameters, up to 200 epochs. Search space: 1–2 layers,
hidden size, dropout 0–0.5, learning rate 10⁻⁴–10⁻² and weight decay 10⁻⁶–10⁻² (both on
a log scale). The best configurations are frozen in `configs/`: LSTM with 199,836
parameters, ConvLSTM with 37,261.

**ConvLSTM, last exam only.** Same configuration and training, but the history is cut
to the last exam. It measures how much of the gain needs the history.

## 5. Evaluation

On the evaluation set (n = 4,378):

- error of a sample = mean |prediction − measured| over the 52 locations (dB);
- MAE of a model = mean error over the samples;
- gain = MAE(persistence) − MAE(model). Positive means better than persistence;
- advantage over the EMA = MAE(EMA) − MAE(model);
- bias = predicted mean sensitivity − measured mean sensitivity. Positive means the model
  predicts a better field than the one measured.

**Confidence intervals.** 95%, from 2,000 bootstrap resamples of patients (seed
20260101), because the two eyes of a patient are not independent. A ratio, such as the
share of the ConvLSTM gain, is computed with both terms from the same resample.

**Progression groups** (results by group). Slope of a line fitted to the MTD against
age. The fit uses the history without its last exam, and never the target. Otherwise a
low value by chance in one of those exams would decide both the group and the error. At
least 4 points are needed, so samples with fewer than 5 exams of history are
"undetermined".

| Group | Rule |
|---|---|
| rapid | slope ≤ −1 dB/year, p < 0.05 |
| moderate | −1 < slope ≤ −0.5 dB/year, p < 0.05 |
| slow | −0.5 < slope ≤ 0 dB/year, p < 0.05 |
| improving | slope > 0, p < 0.05 |
| stable | p ≥ 0.05 |

The rapid group has 41 evaluation samples from 38 patients.
