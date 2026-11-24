"""Step 3 (optional): hyperparameter search with Optuna (TPE), 40 trials per network.

    uv run python scripts/03_tune.py --model lstm
    uv run python scripts/03_tune.py --model convlstm

configs/{lstm,convlstm}.json already hold the values used in the paper; run this only
to redo the search. It overwrites those files. The ConvLSTM search takes ~5 h on a GTX 1660.

Protocol (PROTOCOL.md section 4):
  - only the TRAINING patients of fold 1, split 80/20 into train/validation; the test
    patients of fold 1 never enter the search;
  - same budget for both networks: 40 completed trials, no pruning;
  - every configuration has 1,500 to 200,000 trainable parameters. The range of `hidden`
    allowed for each number of layers is computed before the search, so no trial is
    wasted on a configuration outside the band;
  - objective: best validation L1, with the same `nets.train` used in step 4.

The study is saved in outputs/tune/{model}.db. Rerunning the command resumes it.
"""

import argparse
import json

import numpy as np
import optuna
import torch

from src import config, data, nets

N_TRIALS = 40
DEV_FOLD = 1
MAX_EPOCHS = 200  # the cap used for the paper's search
BAND = (1_500, 200_000)
SPACE = {
    "layers": (1, 2),
    "dropout": (0.0, 0.5),
    "learning_rate": (1e-4, 1e-2),  # log scale
    "weight_decay": (1e-6, 1e-2),  # log scale
}


def hidden_range(name: str, layers: int) -> tuple[int, int]:
    """Smallest and largest `hidden` inside the parameter band (the count grows with hidden)."""
    def count(hidden):
        return nets.n_parameters(name, {"hidden": hidden, "layers": layers, "dropout": 0.0})

    hidden = 1
    while count(hidden) < BAND[0]:
        hidden += 1
    smallest = hidden
    while count(hidden + 1) <= BAND[1]:
        hidden += 1
    return smallest, hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(nets.NETWORKS), required=True)
    args = parser.parse_args()
    name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = data.load_histories()
    t = nets.to_tensors(h, device)
    train_idx = np.flatnonzero(h.fold != DEV_FOLD)
    ranges = {layers: hidden_range(name, layers) for layers in range(SPACE["layers"][0], SPACE["layers"][1] + 1)}
    print(f"=== {name}: hidden range per number of layers {ranges} ===", flush=True)

    def objective(trial: optuna.Trial) -> float:
        layers = trial.suggest_int("layers", *SPACE["layers"])
        hp = {
            "hidden": trial.suggest_int("hidden", *ranges[layers]),
            "layers": layers,
            "dropout": trial.suggest_float("dropout", *SPACE["dropout"]),
            "learning_rate": trial.suggest_float("learning_rate", *SPACE["learning_rate"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *SPACE["weight_decay"], log=True),
        }
        trial.set_user_attr("n_parameters", nets.n_parameters(name, hp))
        _, curve = nets.train(name, hp, t, train_idx, max_epochs=MAX_EPOCHS)
        trial.set_user_attr("epochs", len(curve))
        print(f"  trial {trial.number:2d}  val_l1={min(curve):.4f}  {hp}  "
              f"{trial.user_attrs['n_parameters']} params  {len(curve)} epochs", flush=True)
        return min(curve)

    out_dir = config.OUT_DIR / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{(out_dir / f'{name}.db').as_posix()}",
        load_if_exists=True,
        direction="minimize",
    )
    done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    # A new process restarts the sampler's random generator; shifting the seed by the
    # trials already done avoids proposing the same first points again on resume.
    study.sampler = optuna.samplers.TPESampler(seed=config.SEED + done)
    study.optimize(objective, n_trials=max(0, N_TRIALS - done))

    best = study.best_trial
    hp = {k: best.params[k] for k in ("hidden", "layers", "dropout", "learning_rate", "weight_decay")}
    (config.ROOT / "configs" / f"{name}.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")
    print(f"best: trial {best.number}  val_l1={best.value:.4f}  {best.user_attrs['n_parameters']} params")
    print(f"-> configs/{name}.json")


if __name__ == "__main__":
    main()
