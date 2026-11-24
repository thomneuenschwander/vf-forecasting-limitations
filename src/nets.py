"""The two deep models (LSTM and ConvLSTM) and the training they share.

Both receive, for each exam of the history, the 52 sensitivities and the interval since
the previous exam. After the last exam they also receive the interval to the target and
the patient's age. Both predict the CORRECTION to the last exam (residual formulation):

    prediction = last exam + network(history)

Training is identical for the two, so that the comparison measures the architecture:
AdamW, L1 loss (the MAE itself), batches of 64, early stopping on a validation split of
20% of the training patients. Hyperparameters come from configs/{lstm,convlstm}.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

from src import config
from src.data import Histories

# Fixed scale, not estimated from the data: dB -> (dB - 25) / 8
SCALE_CENTER, SCALE_UNIT = 25.0, 8.0

# Training, identical for both networks
BATCH_SIZE = 64
PATIENCE = 20
MAX_EPOCHS = 600  # the full models stop before epoch 100; only the no-history ConvLSTM goes further
VAL_FRACTION = 0.2

N_IN = config.N_TESTABLE + 1  # 52 sensitivities + interval since the previous exam (years)
N_CONTEXT = 2  # interval to the target (years) and age at the last exam (years / 100)


def normalize(db):
    return (db - SCALE_CENTER) / SCALE_UNIT


def denormalize(z):
    return z * SCALE_UNIT + SCALE_CENTER


def load_hparams(name: str) -> dict:
    return json.loads((config.ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- tensors


@dataclass
class Tensors:
    """All samples on the GPU, in sample_id order. Batches are slices of these."""

    seq: torch.Tensor  # (N, T, 53) normalized, right-padded with 0
    lengths: torch.Tensor  # (N,)
    lengths_host: np.ndarray  # (N,) same, on the CPU (reading it from the GPU would sync)
    context: torch.Tensor  # (N, 2)
    last: torch.Tensor  # (N, 52) normalized
    target: torch.Tensor  # (N, 52) normalized
    pat_id: np.ndarray  # (N,)


def to_tensors(h: Histories, device: torch.device, max_history: int | None = None) -> Tensors:
    """`max_history=1` keeps only the last exam: the network sees no history.

    The kept exams are the LAST ones; the interval of the first kept exam is set to 0,
    as for the first exam of any history.
    """
    n = len(h.n_hist)
    k = h.n_hist if max_history is None else np.minimum(h.n_hist, max_history)
    steps = int(k.max())

    seq = np.zeros((n, steps, N_IN), dtype=np.float32)
    for i in range(n):
        start, end = h.n_hist[i] - k[i], h.n_hist[i]
        gaps = h.gap_months[i, start:end].copy()
        gaps[0] = 0.0
        seq[i, : k[i], : config.N_TESTABLE] = normalize(h.sens[i, start:end])
        seq[i, : k[i], config.N_TESTABLE] = gaps / 12
    context = np.stack([h.delta_t_months / 12, h.age_last / 100], axis=1).astype(np.float32)

    def gpu(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    return Tensors(
        seq=gpu(seq),
        lengths=torch.from_numpy(k.astype(np.int64)).to(device),
        lengths_host=k.astype(np.int64),
        context=gpu(context),
        last=gpu(normalize(h.last)),
        target=gpu(normalize(h.target)),
        pat_id=h.pat_id,
    )


def _batch(t: Tensors, idx: np.ndarray) -> dict:
    """Cut to the longest history IN the batch: the steps after it are padding for all."""
    lengths_host = t.lengths_host[idx]
    steps = int(lengths_host.max())
    gpu_idx = torch.from_numpy(idx).to(t.seq.device, non_blocking=True)
    return {
        "seq": t.seq[:, :steps][gpu_idx],
        "lengths": t.lengths[gpu_idx],
        "lengths_host": lengths_host,
        "context": t.context[gpu_idx],
        "last": t.last[gpu_idx],
        "target": t.target[gpu_idx],
    }


# --------------------------------------------------------------------------- networks


class LSTMNet(nn.Module):
    """Each exam is a vector of 52 values: no information about which points are neighbors."""

    def __init__(self, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            N_IN, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + N_CONTEXT, config.N_TESTABLE)

    def forward(self, b: dict) -> torch.Tensor:
        out, _ = self.rnn(b["seq"])
        # State at the last REAL exam of each sample. The LSTM is causal, so this state
        # never saw the padding: same result as pack_padded_sequence.
        idx = (b["lengths"] - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
        state = out.gather(1, idx).squeeze(1)
        # Dropout on the recurrent state only; the context goes in intact.
        return self.head(torch.cat([self.drop(state), b["context"]], dim=1))


class ConvLSTMCell(nn.Module):
    """LSTM cell whose matrix products are 3x3 convolutions over the 8x9 grid."""

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, 3, padding=1)

    def forward(self, x, h, c):
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMNet(nn.Module):
    """Each exam is a 8x9 map with 3 channels: sensitivity, mask of the 52 valid
    cells, interval since the previous exam. The error is only computed on the 52 cells."""

    def __init__(self, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        grid = pl.read_parquet(config.GRID_PARQUET).filter(pl.col("cluster") != 0).sort("index")
        self.register_buffer("rows", torch.tensor(grid["grid_row"].to_numpy()).long(), persistent=False)
        self.register_buffer("cols", torch.tensor(grid["grid_col"].to_numpy()).long(), persistent=False)
        mask = torch.zeros(config.GRID_SHAPE)
        mask[self.rows, self.cols] = 1.0
        self.register_buffer("mask", mask, persistent=False)

        self.cells = nn.ModuleList(ConvLSTMCell(3 if i == 0 else hidden, hidden) for i in range(layers))
        self.drop = nn.Dropout2d(dropout)  # drops whole channels
        self.head = nn.Conv2d(hidden + N_CONTEXT, 1, 3, padding=1)

    def forward(self, b: dict) -> torch.Tensor:
        seq, n = b["seq"], b["seq"].size(0)
        maps = seq.new_zeros(n, seq.size(1), *config.GRID_SHAPE)  # empty cells = 0 normalized
        maps[:, :, self.rows, self.cols] = seq[:, :, : config.N_TESTABLE]
        gaps = seq[:, :, config.N_TESTABLE]

        # Sort by length (descending): at step k only the first `active` samples still
        # have an exam, so only that prefix is updated. Same as masking, with less work.
        order_host = np.argsort(-b["lengths_host"], kind="stable")
        sorted_lengths = b["lengths_host"][order_host]
        order = torch.from_numpy(order_host).to(seq.device)
        maps, gaps = maps[order], gaps[order]

        mask = self.mask.expand(n, 1, *config.GRID_SHAPE)
        states = [(maps.new_zeros(n, c.hidden, *config.GRID_SHAPE),) * 2 for c in self.cells]
        for step in range(seq.size(1)):
            active = int((sorted_lengths > step).sum())
            if active == 0:
                break
            gap = gaps[:active, step].view(active, 1, 1, 1).expand(active, 1, *config.GRID_SHAPE)
            x = torch.cat([maps[:active, step].unsqueeze(1), mask[:active], gap], dim=1)
            for layer, cell in enumerate(self.cells):
                h, c = states[layer]
                h_new, c_new = cell(x, h[:active], c[:active])
                states[layer] = (torch.cat([h_new, h[active:]]), torch.cat([c_new, c[active:]]))
                x = states[layer][0][:active]

        final = self.drop(states[-1][0][torch.argsort(order)])
        ctx = b["context"].view(n, N_CONTEXT, 1, 1).expand(n, N_CONTEXT, *config.GRID_SHAPE)
        out = self.head(torch.cat([final, ctx], dim=1)).squeeze(1)
        return out[:, self.rows, self.cols]


NETWORKS = {"lstm": LSTMNet, "convlstm": ConvLSTMNet}


def build(name: str, hp: dict) -> nn.Module:
    return NETWORKS[name](hidden=hp["hidden"], layers=hp["layers"], dropout=hp["dropout"])


def n_parameters(name: str, hp: dict) -> int:
    return sum(p.numel() for p in build(name, hp).parameters() if p.requires_grad)


# --------------------------------------------------------------------------- training


def _l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def _predict_batch(net: nn.Module, b: dict) -> torch.Tensor:
    return b["last"] + net(b)  # residual formulation


def split_validation(pat_id: np.ndarray, idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """20% of the training PATIENTS go to validation (both eyes together)."""
    patients = np.random.default_rng(seed).permutation(np.unique(pat_id[idx]))
    val_patients = patients[: max(1, round(len(patients) * VAL_FRACTION))]
    in_val = np.isin(pat_id[idx], val_patients)
    return idx[~in_val], idx[in_val]


def train(
    name: str,
    hp: dict,
    t: Tensors,
    train_idx: np.ndarray,
    seed: int = config.SEED,
    max_epochs: int = MAX_EPOCHS,
    on_epoch=None,
) -> tuple[nn.Module, list[float]]:
    """Train on `train_idx` (positions in sample_id order). Returns the network at its
    best validation epoch and the validation L1 of every epoch.

    `on_epoch(epoch, val_l1)` lets the hyperparameter search stop a bad trial early.
    """
    torch.manual_seed(seed)
    fit_idx, val_idx = split_validation(t.pat_id, train_idx, seed)
    device = t.seq.device
    net = build(name, hp).to(device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"], fused=device.type == "cuda"
    )

    val = _batch(t, val_idx)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, curve = float("inf"), -1, None, []
    for epoch in range(max_epochs):
        net.train()
        order = rng.permutation(len(fit_idx))
        for start in range(0, len(order), BATCH_SIZE):
            b = _batch(t, fit_idx[order[start : start + BATCH_SIZE]])
            opt.zero_grad(set_to_none=True)
            _l1(_predict_batch(net, b), b["target"]).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            loss = _l1(_predict_batch(net, val), val["target"]).item()
        curve.append(loss)
        if on_epoch is not None:
            on_epoch(epoch, loss)

        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{name}: validation loss never improved (diverged?)")
    net.load_state_dict(best_state)
    return net.eval(), curve


def predict(net: nn.Module, t: Tensors, idx: np.ndarray) -> np.ndarray:
    """(len(idx), 52) in dB."""
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), 512):
            out.append(_predict_batch(net, _batch(t, idx[start : start + 512])))
    return denormalize(torch.cat(out).cpu().numpy().astype(np.float64))
