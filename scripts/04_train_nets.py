"""Step 4: train the LSTM and the ConvLSTM on the 5 folds (GPU).

    uv run python scripts/04_train_nets.py                       # both networks
    uv run python scripts/04_train_nets.py --model convlstm --max-history 1

For each fold k the network is trained on the patients of the other 4 folds and predicts
the patients of fold k. The union of the 5 test folds gives one out-of-fold prediction per
sample. Hyperparameters come from configs/ (step 3, or as frozen for the paper).

Writes outputs/preds/{model}.parquet, or {model}_h{K}.parquet with --max-history K,
and outputs/train_log/{same name}.json with the epochs of each fold.

Takes about 35 min (ConvLSTM), 2 min (LSTM) and 12 min (ConvLSTM without history) on a
GTX 1660. GPU training is not bit-for-bit deterministic: expect MAE within ~0.01 dB.
"""

import argparse
import json
import time

import numpy as np
import torch

from src import config, data, nets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(nets.NETWORKS), action="append")
    parser.add_argument("--max-history", type=int, default=None, help="keep only the last K exams")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, config.N_FOLDS + 1)))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = data.load_histories()
    t = nets.to_tensors(h, device, args.max_history)

    for name in args.model or ["lstm", "convlstm"]:
        hp = nets.load_hparams(name)
        out_name = name if args.max_history is None else f"{name}_h{args.max_history}"
        print(f"=== {out_name}  {nets.n_parameters(name, hp)} parameters  {device} ===", flush=True)

        pred = np.full_like(h.target, np.nan)
        log = []
        for fold in args.folds:
            started = time.perf_counter()
            train_idx = np.flatnonzero(h.fold != fold)
            test_idx = np.flatnonzero(h.fold == fold)
            net, curve = nets.train(name, hp, t, train_idx)
            pred[test_idx] = nets.predict(net, t, test_idx)

            seconds = time.perf_counter() - started
            log.append({"fold": fold, "epochs": len(curve), "best_val_l1": min(curve), "seconds": round(seconds)})
            mae = np.abs(np.clip(pred[test_idx], 0, 50) - h.target[test_idx])[h.is_last[test_idx]].mean()
            print(f"  fold {fold}  epochs={len(curve):3d}  val_l1={min(curve):.4f}  "
                  f"test MAE={mae:.3f}  {seconds:5.0f}s", flush=True)

        if len(args.folds) < config.N_FOLDS:
            print("  (not all folds: nothing written)")
            continue
        data.write_predictions(out_name, h, pred)
        log_dir = config.OUT_DIR / "train_log"
        log_dir.mkdir(exist_ok=True)
        (log_dir / f"{out_name}.json").write_text(json.dumps({"hparams": hp, "folds": log}, indent=2))
        mae = np.abs(data.read_predictions(out_name) - h.target)[h.is_last].mean()
        print(f"{out_name}: MAE on the evaluation set {mae:.3f} dB", flush=True)


if __name__ == "__main__":
    main()
