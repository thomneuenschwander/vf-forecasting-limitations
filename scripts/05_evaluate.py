"""Step 5: every number of the Results section, from the saved predictions (no GPU).

    uv run python scripts/05_evaluate.py

Writes to results/ (versioned, small):
    main.csv         per model: MAE, gain over persistence, % of the ConvLSTM gain,
                     advantage over the EMA, bias                         (Table 2, obj. ii)
    by_history.csv   the same, by number of exams in the history           (Figure 3b, obj. iii)
    rapid.csv        eyes that get worse fast: gain and bias               (Table 3, obj. iv)
    groups.csv       how many evaluation samples fall in each progression group
    by_fold.csv      ConvLSTM gain in each fold (check of the hyperparameter search)

All on the evaluation set (last sample of each series, n = 4,378), with 95% CIs from
2,000 bootstrap resamples of patients.
"""

import json

import numpy as np
import polars as pl

from src import config, data, nets
from src.evaluate import GROUPS, PatientBootstrap, bias, progression_group, sample_error

MODELS = ["persistence", "plr", "ema", "lstm", "convlstm", "convlstm_h1"]
REFERENCE = "persistence"  # gain = MAE(persistence) - MAE(model)
SMOOTHER = "ema"  # advantage = MAE(ema) - MAE(model)
BEST = "convlstm"  # "% of the gain" is relative to this one
HISTORY_BINS = [("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), (">=6", 6, 99), (">=3", 3, 99)]
RESULTS = config.ROOT / "results"


def ci(prefix: str, estimate: tuple[float, float, float]) -> dict:
    value, low, high = estimate
    return {prefix: value, f"{prefix}_low": low, f"{prefix}_high": high}


def n_parameters(model: str) -> int:
    if model in ("persistence", "ema"):
        return 0
    if model == "plr":
        return None  # a line per location, fitted in each sample
    return nets.n_parameters(model.removesuffix("_h1"), nets.load_hparams(model.removesuffix("_h1")))


def rows_for(mask: np.ndarray, err: dict, pred: dict, h: data.Histories, models=MODELS, **extra) -> list[dict]:
    boot = PatientBootstrap(h.pat_id[mask])
    best_gain = err[REFERENCE][mask] - err[BEST][mask]
    rows = []
    for m in models:
        gain = err[REFERENCE][mask] - err[m][mask]
        rows.append({
            **extra,
            "model": m,
            "n": int(mask.sum()),
            "patients": len(np.unique(h.pat_id[mask])),
            **ci("mae", boot.mean(err[m][mask])),
            **ci("gain", boot.mean(gain)),
            **ci("share_of_convlstm_gain", boot.ratio(gain, best_gain)),
            **ci("advantage_over_ema", boot.mean(err[SMOOTHER][mask] - err[m][mask])),
            **ci("bias", boot.mean(bias(pred[m][mask], h.target[mask]))),
        })
    return rows


def main() -> None:
    h = data.load_histories()
    pred = {m: data.read_predictions(m) for m in MODELS}
    err = {m: sample_error(p, h.target) for m, p in pred.items()}
    last = h.is_last
    RESULTS.mkdir(exist_ok=True)

    # Table 2
    main_rows = rows_for(last, err, pred, h)
    for row in main_rows:
        row["parameters"] = n_parameters(row["model"])
    pl.DataFrame(main_rows).write_csv(RESULTS / "main.csv")

    # Figure 3b and the pLR paragraph
    by_history = []
    for label, low, high in HISTORY_BINS:
        mask = last & (h.n_hist >= low) & (h.n_hist <= high)
        for row in rows_for(mask, err, pred, h, history=label):
            row["share_of_samples"] = mask.sum() / last.sum()
            by_history.append(row)
    pl.DataFrame(by_history).write_csv(RESULTS / "by_history.csv")

    # Table 3
    group = progression_group()
    rapid = last & (group == "rapid")
    pl.DataFrame(rows_for(rapid, err, pred, h, models=MODELS[:5], group="rapid")).write_csv(RESULTS / "rapid.csv")

    counts = [{"group": g, "n": int((last & (group == g)).sum())} for g in GROUPS]
    defined = sum(c["n"] for c in counts if c["group"] != "undetermined")
    for c in counts:
        c["share_of_defined"] = c["n"] / defined if c["group"] != "undetermined" else None
    pl.DataFrame(counts).write_csv(RESULTS / "groups.csv")

    # Section 3.3: the search used fold 1; is its gain out of line with the others?
    by_fold = [
        {"fold": f, "convlstm_gain": float((err[REFERENCE] - err[BEST])[last & (h.fold == f)].mean())}
        for f in range(1, config.N_FOLDS + 1)
    ]
    pl.DataFrame(by_fold).write_csv(RESULTS / "by_fold.csv")

    print_summary(main_rows, by_history, rapid, h, counts, by_fold, err, pred)


def fmt(row: dict, key: str) -> str:
    return f"{row[key]:.2f} [{row[key + '_low']:.2f}; {row[key + '_high']:.2f}]"


def print_summary(main_rows, by_history, rapid, h, counts, by_fold, err, pred) -> None:
    rows = {r["model"]: r for r in main_rows}
    print("Evaluation set: n = 4,378 samples\n")
    print(f"{'model':12s} {'MAE':>20s} {'gain':>22s} {'% of ConvLSTM gain':>20s} {'vs EMA':>22s}")
    for m in MODELS:
        r = rows[m]
        share = f"{r['share_of_convlstm_gain']:.0%} [{r['share_of_convlstm_gain_low']:.0%}; {r['share_of_convlstm_gain_high']:.0%}]"
        print(f"{m:12s} {fmt(r, 'mae'):>20s} {fmt(r, 'gain'):>22s} {share:>20s} {fmt(r, 'advantage_over_ema'):>22s}")

    print("\nBy exams in the history: gain (pLR) | advantage over EMA (LSTM, ConvLSTM)")
    for label, *_ in HISTORY_BINS:
        r = {x["model"]: x for x in by_history if x["history"] == label}
        print(f"  {label:4s} {r['plr']['share_of_samples']:4.0%} of samples   pLR gain {fmt(r['plr'], 'gain'):>22s}   "
              f"LSTM {fmt(r['lstm'], 'advantage_over_ema'):>22s}   ConvLSTM {fmt(r['convlstm'], 'advantage_over_ema'):>22s}")

    print(f"\nRapid group: {int(rapid.sum())} samples, {len(np.unique(h.pat_id[rapid]))} patients")
    boot = PatientBootstrap(h.pat_id[rapid])
    for m in MODELS[:5]:
        g = boot.mean(err["persistence"][rapid] - err[m][rapid])
        b = boot.mean(bias(pred[m][rapid], h.target[rapid]))
        print(f"  {m:12s} gain {g[0]:.2f} [{g[1]:.2f}; {g[2]:.2f}]   bias {b[0]:.2f} [{b[1]:.2f}; {b[2]:.2f}]")
    print("  groups among defined samples: " + ", ".join(
        f"{c['group']} {c['share_of_defined']:.0%}" for c in counts if c["share_of_defined"] is not None))

    print("\nConvLSTM gain by fold: " + "  ".join(f"{r['fold']}: {r['convlstm_gain']:.2f}" for r in by_fold))
    print(f"\n-> {RESULTS}")


if __name__ == "__main__":
    main()
