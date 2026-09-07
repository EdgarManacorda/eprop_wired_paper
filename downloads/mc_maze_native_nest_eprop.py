#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MC_Maze native NEST e-prop tutorial
===================================

This script trains a recurrent spiking neural network on MC_Maze neural spike
trains using native NEST e-prop components.

The network receives a 1500 ms spike raster from recorded units and uses a
three-segment temporal readout: 8 classes x 3 time segments = 24 readout neurons.
The final class prediction is obtained by averaging class scores across segments.

Run from the project root:
    python scripts/mc_maze_native_nest_eprop.py

Expected prepared data location:
    data/prepared/mc_maze_maze_id_8classes_move_onset_1500ms_240units

Main outputs:
    results/mc_maze/mc_maze_native_nest_eprop_tutorial/
        config.json
        final_metrics.json
        logs/*.csv
        figures/*.png
        weights/*.npy
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nest
except Exception as exc:
    raise RuntimeError("Run this script inside the NEST Python environment.") from exc


# =============================================================================
# Parameters
# =============================================================================

# The script is expected to live in a scripts/ folder. If it is run from another
# location, paths are resolved relative to the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else Path.cwd()

PREPARED_DIR = PROJECT_ROOT / "data/prepared/mc_maze_maze_id_8classes_move_onset_1500ms_240units"
RESULTS_ROOT = PROJECT_ROOT / "results/mc_maze"
RUN_NAME = "mc_maze_native_nest_eprop_tutorial"

# Main parameters readers may want to change first:
#   N_REC, N_ITER_TRAIN, GROUP_SIZE, ETA_TRAIN, TARGET_AMPLITUDE,
#   N_READOUT_SEGMENTS, sparsity values, and the input/background weight scales.
N_REC = 150
N_ITER_TRAIN = 150
GROUP_SIZE = 16                 # 8 classes -> 2 samples/class/group
ETA_TRAIN = 2e-7
ETA_EVAL = 0.0
SEQUENCE_MS = 1500.0
N_READOUT_SEGMENTS = 3
SEED = 123

RESOLUTION_MS = 1.0
THREADS = 1
TARGET_AMPLITUDE = 0.7

# Native e-prop optimizer bounds.
WMIN = -3.0
WMAX = 3.0
OPTIMIZER_BATCH_SIZE = 1
OPTIMIZE_EACH_STEP = False

# Connectivity sparsity.
KEEP_IN_REC = 0.15
KEEP_REC_REC = 0.05
KEEP_REC_OUT = 1.0

# Initial weight scales.
W_IN_SCALE = 0.25
W_REC_SCALE = 0.8
W_OUT_SCALE = 1.0
FEEDBACK_SCALE = 0.2

# Optional background drive for the recurrent population.
BACKGROUND_RATE_HZ = 50.0
BACKGROUND_WEIGHT = 0.05

# Evaluation groups.
N_TRAIN_EVAL_GROUPS = 6          # 6 groups x 16 = 96 train samples
N_VAL_GROUPS = 6                 # 6 groups x 16 = 96 validation samples

# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def node_ids(nodes) -> np.ndarray:
    if hasattr(nodes, "tolist"):
        return np.array(nodes.tolist(), dtype=int)
    return np.array(list(nodes), dtype=int)


def quiet_nest() -> None:
    nest.set_verbosity("M_WARNING")


def filter_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    valid = set(nest.GetDefaults(model_name).keys())
    return {k: v for k, v in params.items() if k in valid}


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# =============================================================================
# Plotting
# =============================================================================

def plot_training_curves(log_df: pd.DataFrame, figures_dir: Path) -> None:
    train_df = log_df[log_df["phase"] == "train"]
    train_eval_df = log_df[log_df["phase"] == "train_eval"]
    val_df = log_df[log_df["phase"] == "val"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["accuracy"], marker="o", label="train")
    ax.scatter(train_eval_df["global_step"], train_eval_df["accuracy"], marker="x", s=80, label="train_eval")
    ax.plot(val_df["global_step"], val_df["accuracy"], marker="o", label="val")
    ax.set_title("MC_Maze native NEST e-prop accuracy")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["loss"], marker="o", label="train")
    ax.scatter(train_eval_df["global_step"], train_eval_df["loss"], marker="x", s=80, label="train_eval")
    ax.plot(val_df["global_step"], val_df["loss"], marker="o", label="val")
    ax.set_title("MC_Maze native NEST e-prop loss")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("MSE-like loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["recurrent_rate_hz"], marker="o", label="train")
    ax.plot(train_eval_df["global_step"], train_eval_df["recurrent_rate_hz"], marker="x", label="train_eval")
    ax.plot(val_df["global_step"], val_df["recurrent_rate_hz"], marker="o", label="val")
    ax.set_title("Recurrent firing rate")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("Mean recurrent firing rate [Hz]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "recurrent_firing_rate.png", dpi=200)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_weight_matrix(W: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Pre neuron / feature index")
    ax.set_ylabel("Post neuron / readout index")
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# Data loading and preprocessing
# =============================================================================

@dataclass
class Sample:
    sample_id: str
    original_label: int
    class_index: int
    spikes_by_channel: list[np.ndarray]


def load_metadata(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df = pd.read_csv(prepared_dir / "train_metadata.csv")
    val_df = pd.read_csv(prepared_dir / "val_metadata.csv")

    label_col = "class_id"
    train_df["class_id_resolved"] = train_df[label_col].astype(int)
    val_df["class_id_resolved"] = val_df[label_col].astype(int)

    config_path = prepared_dir / "config.json"
    prepared_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            prepared_config = json.load(f)

    return train_df, val_df, prepared_config


def find_sample_path(row: pd.Series, prepared_dir: Path) -> Path:
    p = Path(str(row["sample_file"]))
    return p if p.is_absolute() else prepared_dir / p


def load_npz_as_spike_channels(sample_path: Path, sequence_ms: float, n_in: int | None = None) -> tuple[list[np.ndarray], int]:
    data = np.load(sample_path, allow_pickle=True)

    units = np.asarray(data["channels"], dtype=int)
    times = np.asarray(data["times_ms"], dtype=float)

    if n_in is None:
        n_in = int(np.asarray(data["binned_counts"]).shape[1])

    mask = (
        np.isfinite(times)
        & (times >= 0.0)
        & (times <= sequence_ms)
        & (units >= 0)
        & (units < n_in)
    )
    times = np.round(np.clip(times[mask], 1.0, sequence_ms)).astype(float)
    units = units[mask]

    key = units.astype(np.int64) * 10_000_000 + times.astype(np.int64)
    _, unique_idx = np.unique(key, return_index=True)
    times = times[unique_idx]
    units = units[unique_idx]

    spikes_by_channel = []
    for u in range(n_in):
        spikes_by_channel.append(np.sort(times[units == u]).astype(float))

    return spikes_by_channel, n_in


def estimate_n_in(prepared_dir: Path, train_df: pd.DataFrame) -> int:
    sample_path = find_sample_path(train_df.iloc[0], prepared_dir)
    _, n_in = load_npz_as_spike_channels(sample_path, sequence_ms=SEQUENCE_MS, n_in=None)
    return n_in


def load_samples(df: pd.DataFrame, prepared_dir: Path, n_in: int, split_name: str) -> list[Sample]:
    samples = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        sample_path = find_sample_path(row, prepared_dir)
        spikes_by_channel, _ = load_npz_as_spike_channels(sample_path, sequence_ms=SEQUENCE_MS, n_in=n_in)
        class_id = int(row["class_id_resolved"])
        raw_label = int(row["raw_label"]) if "raw_label" in row.index and pd.notna(row["raw_label"]) else class_id
        samples.append(
            Sample(
                sample_id=f"{split_name}_{row_index}_{sample_path.stem}",
                original_label=raw_label,
                class_index=class_id,
                spikes_by_channel=spikes_by_channel,
            )
        )
    return samples


class BalancedGroupLoader:
    def __init__(self, samples: list[Sample], n_classes: int, group_size: int, rng: np.random.Generator):
        self.samples_by_class = {c: [] for c in range(n_classes)}
        for s in samples:
            self.samples_by_class[s.class_index].append(s)

        self.n_classes = n_classes
        self.group_size = group_size
        self.per_class = group_size // n_classes
        self.rng = rng
        self.permutations = {c: self.rng.permutation(len(self.samples_by_class[c])) for c in range(n_classes)}
        self.ptr = {c: 0 for c in range(n_classes)}

    def _take_one(self, c: int) -> Sample:
        if self.ptr[c] >= len(self.samples_by_class[c]):
            self.permutations[c] = self.rng.permutation(len(self.samples_by_class[c]))
            self.ptr[c] = 0
        idx = int(self.permutations[c][self.ptr[c]])
        self.ptr[c] += 1
        return self.samples_by_class[c][idx]

    def get_group(self) -> list[Sample]:
        group = []
        for c in range(self.n_classes):
            for _ in range(self.per_class):
                group.append(self._take_one(c))
        self.rng.shuffle(group)
        return group


# =============================================================================
# NEST e-prop helpers
# =============================================================================

def set_eprop_optimizer(eta: float) -> None:
    nest.SetDefaults(
        "eprop_synapse",
        {
            "optimizer": {
                "type": "gradient_descent",
                "eta": float(eta),
                "batch_size": OPTIMIZER_BATCH_SIZE,
                "optimize_each_step": OPTIMIZE_EACH_STEP,
                "Wmin": WMIN,
                "Wmax": WMAX,
            }
        },
    )


def sparse_mask(W: np.ndarray, keep_probability: float, rng: np.random.Generator) -> np.ndarray:
    return W * (rng.random(W.shape) < keep_probability)


def glorot(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3.0 * scale)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def connect_nonzero(W_post_pre: np.ndarray, pre, post, synapse_model: str, delay_ms: float) -> None:
    post_idx, pre_idx = np.nonzero(np.abs(W_post_pre) > 0.0)
    pre_ids = node_ids(pre)
    post_ids = node_ids(post)

    nest.Connect(
        pre_ids[pre_idx].tolist(),
        post_ids[post_idx].tolist(),
        {"rule": "one_to_one"},
        {
            "synapse_model": synapse_model,
            "weight": W_post_pre[post_idx, pre_idx].astype(float).tolist(),
            "delay": [float(delay_ms)] * len(pre_idx),
        },
    )


def get_weight_matrix(pre, post) -> np.ndarray:
    pre_ids = node_ids(pre)
    post_ids = node_ids(post)
    pre_map = {int(gid): i for i, gid in enumerate(pre_ids)}
    post_map = {int(gid): i for i, gid in enumerate(post_ids)}
    W = np.zeros((len(post_ids), len(pre_ids)), dtype=float)

    conns = nest.GetConnections(pre, post)
    values = conns.get(["source", "target", "weight"])
    for source, target, weight in zip(values["source"], values["target"], values["weight"]):
        W[post_map[int(target)], pre_map[int(source)]] = float(weight)

    return W


def clear_recorder_buffers(*recorders) -> None:
    for rec in recorders:
        rec.set({"n_events": 0})


def add_step(times: list[float], values: list[float], t: float, v: float) -> None:
    t = float(t)
    v = float(v)
    if times and abs(times[-1] - t) < 1e-9:
        values[-1] = v
    else:
        times.append(t)
        values.append(v)


# =============================================================================
# Training group construction and evaluation
# =============================================================================

def build_generators_for_group(
    group: list[Sample],
    current_time_ms: float,
    n_classes: int,
    n_readouts: int,
    steps: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, list[str], np.ndarray]:
    spike_times_by_input = [[] for _ in range(N_IN)]
    y_true = []
    sample_ids = []
    original_labels = []
    segment_ms = SEQUENCE_MS / N_READOUT_SEGMENTS

    for gi, sample in enumerate(group):
        sample_start = current_time_ms + gi * SEQUENCE_MS
        input_offset = sample_start + steps["offset_gen"]

        y_true.append(sample.class_index)
        sample_ids.append(sample.sample_id)
        original_labels.append(sample.original_label)

        for ch, times in enumerate(sample.spikes_by_channel):
            abs_times = input_offset + times
            abs_times = abs_times[abs_times > current_time_ms]
            spike_times_by_input[ch].extend(abs_times.tolist())

    spike_gen_params = [
        {"spike_times": sorted(np.unique(np.asarray(times, dtype=float)).tolist())}
        for times in spike_times_by_input
    ]

    target_params = []
    window_params = []

    for out_idx in range(n_readouts):
        segment = out_idx // n_classes
        cls = out_idx % n_classes
        target_times = []
        target_values = []
        window_times = []
        window_values = []

        for gi, true_cls in enumerate(y_true):
            sample_start = current_time_ms + steps["total_offset"] + gi * SEQUENCE_MS
            segment_start = sample_start + segment * segment_ms
            segment_end = sample_start + (segment + 1) * segment_ms

            target_value = TARGET_AMPLITUDE if int(true_cls) == cls else 0.0

            add_step(target_times, target_values, sample_start, 0.0)
            add_step(target_times, target_values, segment_start, target_value)
            add_step(target_times, target_values, segment_end, 0.0)

            add_step(window_times, window_values, sample_start, 0.0)
            add_step(window_times, window_values, segment_start, 1.0)
            add_step(window_times, window_values, segment_end, 0.0)

        target_params.append(
            {
                "amplitude_times": np.asarray(target_times, dtype=float),
                "amplitude_values": np.asarray(target_values, dtype=float),
            }
        )

        window_params.append(
            {
                "amplitude_times": np.asarray(window_times, dtype=float),
                "amplitude_values": np.asarray(window_values, dtype=float),
            }
        )

    return (
        spike_gen_params,
        target_params,
        window_params,
        np.asarray(y_true, dtype=int),
        sample_ids,
        np.asarray(original_labels, dtype=int),
    )


def evaluate_readout(
    multimeter,
    readout_nodes,
    current_time_ms: float,
    n_classes: int,
    group_size: int,
    steps: dict[str, float],
    y_true: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    out_ids = node_ids(readout_nodes)
    events = multimeter.get("events")

    times = np.asarray(events["times"], dtype=float)
    senders = np.asarray(events["senders"], dtype=int)
    readout_signal = np.asarray(events["readout_signal"], dtype=float)

    segment_ms = SEQUENCE_MS / N_READOUT_SEGMENTS
    y_pred = []
    scores_all = []
    losses = []

    for gi in range(group_size):
        class_scores = np.zeros(n_classes, dtype=float)
        flat_scores = []
        flat_targets = []

        sample_start = current_time_ms + steps["total_offset"] + gi * SEQUENCE_MS

        for segment in range(N_READOUT_SEGMENTS):
            win_start = sample_start + segment * segment_ms
            win_end = sample_start + (segment + 1) * segment_ms

            for cls in range(n_classes):
                out_idx = segment * n_classes + cls
                gid = int(out_ids[out_idx])
                mask = (times >= win_start) & (times < win_end) & (senders == gid)
                vals = readout_signal[mask]
                score = float(np.mean(vals)) if len(vals) > 0 else 0.0

                class_scores[cls] += score / N_READOUT_SEGMENTS
                flat_scores.append(score)
                flat_targets.append(TARGET_AMPLITUDE if int(y_true[gi]) == cls else 0.0)

        y_pred.append(int(np.argmax(class_scores)))
        scores_all.append(class_scores)
        losses.append(0.5 * float(np.mean((np.asarray(flat_scores) - np.asarray(flat_targets)) ** 2)))

    y_pred = np.asarray(y_pred, dtype=int)
    scores_all = np.asarray(scores_all, dtype=float)
    accuracy = float(np.mean(y_pred == y_true))
    loss = float(np.mean(losses))

    return accuracy, loss, y_pred, scores_all


def recurrent_firing_rate(spike_recorder, current_time_ms: float, run_duration_ms: float, n_rec: int) -> float:
    events = spike_recorder.get("events")
    times = np.asarray(events["times"], dtype=float)
    n_spikes = int(np.sum((times >= current_time_ms) & (times < current_time_ms + run_duration_ms)))
    return float(n_spikes / n_rec / max(1e-9, run_duration_ms / 1000.0))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global N_IN

    t0 = time.time()
    rng = set_global_seed(SEED)

    output_dir = RESULTS_ROOT / RUN_NAME
    figures_dir = output_dir / "figures"
    weights_dir = output_dir / "weights"
    logs_dir = output_dir / "logs"

    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    ensure_dir(weights_dir)
    ensure_dir(logs_dir)

    print("\n=== Native NEST e-prop on MC_Maze ===")
    print(f"Prepared dir:       {PREPARED_DIR}")
    print(f"Output dir:         {output_dir}")
    print(f"N recurrent:        {N_REC}")
    print(f"Sequence ms:        {SEQUENCE_MS}")
    print(f"Readout segments:   {N_READOUT_SEGMENTS}")
    print(f"Group size:         {GROUP_SIZE}")
    print(f"Train iter:         {N_ITER_TRAIN}")
    print(f"eta_train:          {ETA_TRAIN}")
    print(f"background:         {BACKGROUND_RATE_HZ} Hz, weight {BACKGROUND_WEIGHT}")

    train_df, val_df, prepared_config = load_metadata(PREPARED_DIR)
    n_classes = int(max(train_df["class_id_resolved"].max(), val_df["class_id_resolved"].max()) + 1)
    N_IN = estimate_n_in(PREPARED_DIR, train_df)
    n_readouts = n_classes * N_READOUT_SEGMENTS

    print("\nDataset summary:")
    print(f"  Train samples:     {len(train_df)}")
    print(f"  Val samples:       {len(val_df)}")
    print(f"  N input units:     {N_IN}")
    print(f"  N recurrent units: {N_REC}")
    print(f"  N classes:         {n_classes}")
    print(f"  N readouts:        {n_readouts} ({N_READOUT_SEGMENTS} x {n_classes})")

    print("\nLoading samples into memory...")
    train_samples = load_samples(train_df, PREPARED_DIR, N_IN, "train")
    val_samples = load_samples(val_df, PREPARED_DIR, N_IN, "val")

    train_loader = BalancedGroupLoader(train_samples, n_classes, GROUP_SIZE, rng)
    train_eval_loader = BalancedGroupLoader(train_samples, n_classes, GROUP_SIZE, rng)
    val_loader = BalancedGroupLoader(val_samples, n_classes, GROUP_SIZE, rng)

    train_spike_counts = np.asarray([sum(len(x) for x in s.spikes_by_channel) for s in train_samples])
    val_spike_counts = np.asarray([sum(len(x) for x in s.spikes_by_channel) for s in val_samples])
    print("\nSpike count after preprocessing:")
    print(f"  train mean={train_spike_counts.mean():.1f}, median={np.median(train_spike_counts):.1f}, max={train_spike_counts.max()}")
    print(f"  val   mean={val_spike_counts.mean():.1f}, median={np.median(val_spike_counts):.1f}, max={val_spike_counts.max()}")

    # ------------------------------------------------------------------
    # NEST setup
    # ------------------------------------------------------------------
    nest.ResetKernel()
    nest.set(resolution=RESOLUTION_MS, print_time=False, total_num_virtual_procs=THREADS)
    quiet_nest()

    steps = {
        "sequence": SEQUENCE_MS,
        "offset_gen": RESOLUTION_MS,
        "delay_in_rec": RESOLUTION_MS,
        "extension_sim": RESOLUTION_MS,
        "final_update": 3.0 * RESOLUTION_MS,
    }
    steps["total_offset"] = steps["offset_gen"] + steps["delay_in_rec"]

    kappa = 0.98
    kappa_reg = 0.98
    scale_factor = 1.0 - kappa

    rec_params = filter_model_params(
        "eprop_iaf",
        {
            "C_m": 1.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "V_m": 0.0,
            "V_th": 0.6,
            "t_ref": 1.0,
            "tau_m": 30.0,
            "eprop_isi_trace_cutoff": 100.0,
            "f_target": 10.0,
            "c_reg": 2.0 / SEQUENCE_MS / (scale_factor ** 2),
            "kappa": kappa,
            "kappa_reg": kappa_reg,
            "surrogate_gradient_function": "piecewise_linear",
            "gamma": 0.3,
            "beta": 1.0,
        },
    )

    out_params = filter_model_params(
        "eprop_readout",
        {
            "C_m": 1.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "V_m": 0.0,
            "tau_m": 100.0,
            "eprop_isi_trace_cutoff": 100.0,
        },
    )

    print("\nCreating NEST native e-prop network...")
    gen_spk_in = nest.Create("spike_generator", N_IN)
    nrns_in = nest.Create("parrot_neuron", N_IN)
    nrns_rec = nest.Create("eprop_iaf", N_REC, rec_params)
    nrns_out = nest.Create("eprop_readout", n_readouts, out_params)

    gen_rate_target = nest.Create("step_rate_generator", n_readouts)
    gen_learning_window = nest.Create("step_rate_generator", n_readouts)

    mm_out = nest.Create(
        "multimeter",
        {
            "interval": RESOLUTION_MS,
            "record_from": ["readout_signal", "target_signal", "error_signal"],
            "start": 0.0,
            "label": "multimeter_out",
        },
    )
    sr_rec = nest.Create("spike_recorder", {"start": 0.0, "label": "spike_recorder_rec"})

    bg = nest.Create("poisson_generator", params={"rate": float(BACKGROUND_RATE_HZ)})

    # ------------------------------------------------------------------
    # Weights and connections
    # ------------------------------------------------------------------
    print("Initializing weights...")

    W_in_rec = rng.uniform(0.5 * W_IN_SCALE, 1.5 * W_IN_SCALE, size=(N_REC, N_IN))
    W_in_rec = sparse_mask(W_in_rec, KEEP_IN_REC, rng)

    W_rec_rec = rng.normal(0.0, W_REC_SCALE / math.sqrt(N_REC), size=(N_REC, N_REC))
    np.fill_diagonal(W_rec_rec, 0.0)
    W_rec_rec = sparse_mask(W_rec_rec, KEEP_REC_REC, rng)
    np.fill_diagonal(W_rec_rec, 0.0)

    W_rec_out = glorot(N_REC, n_readouts, rng).T * W_OUT_SCALE * scale_factor
    W_rec_out = sparse_mask(W_rec_out, KEEP_REC_OUT, rng)

    W_feedback = rng.normal(0.0, FEEDBACK_SCALE, size=(N_REC, n_readouts))

    print(f"  W_in_rec nonzero:  {np.count_nonzero(W_in_rec)} / {W_in_rec.size}")
    print(f"  W_rec_rec nonzero: {np.count_nonzero(W_rec_rec)} / {W_rec_rec.size}")
    print(f"  W_rec_out nonzero: {np.count_nonzero(W_rec_out)} / {W_rec_out.size}")

    set_eprop_optimizer(ETA_TRAIN)

    print("Connecting network...")
    static_1ms = {"synapse_model": "static_synapse", "delay": steps["delay_in_rec"], "weight": 1.0}

    nest.Connect(gen_spk_in, nrns_in, {"rule": "one_to_one"}, static_1ms)
    nest.Connect(bg, nrns_rec, "all_to_all", {"synapse_model": "static_synapse", "weight": BACKGROUND_WEIGHT, "delay": 1.0})

    connect_nonzero(W_in_rec, nrns_in, nrns_rec, "eprop_synapse", steps["delay_in_rec"])
    connect_nonzero(W_rec_rec, nrns_rec, nrns_rec, "eprop_synapse", steps["delay_in_rec"])
    connect_nonzero(W_rec_out, nrns_rec, nrns_out, "eprop_synapse", steps["delay_in_rec"])

    nest.Connect(
        nrns_out,
        nrns_rec,
        {"rule": "all_to_all", "allow_autapses": False},
        {
            "synapse_model": "eprop_learning_signal_connection",
            "delay": steps["delay_in_rec"],
            "weight": W_feedback,
        },
    )

    nest.Connect(
        gen_rate_target,
        nrns_out,
        {"rule": "one_to_one"},
        {
            "synapse_model": "rate_connection_delayed",
            "delay": steps["delay_in_rec"],
            "receptor_type": 2,
        },
    )

    nest.Connect(
        gen_learning_window,
        nrns_out,
        {"rule": "one_to_one"},
        {
            "synapse_model": "rate_connection_delayed",
            "delay": steps["delay_in_rec"],
            "receptor_type": 1,
        },
    )

    nest.Connect(mm_out, nrns_out, {"rule": "all_to_all"}, static_1ms)
    nest.Connect(nrns_rec, sr_rec, {"rule": "all_to_all"}, static_1ms)

    gen_spk_final_update = nest.Create("spike_generator", 1)
    nest.Connect(
        gen_spk_final_update,
        nrns_in + nrns_rec,
        "all_to_all",
        {"synapse_model": "static_synapse", "weight": 1000.0, "delay": steps["delay_in_rec"]},
    )

    print("Reading initial weights...")
    W0_in_rec = get_weight_matrix(nrns_in, nrns_rec)
    W0_rec_rec = get_weight_matrix(nrns_rec, nrns_rec)
    W0_rec_out = get_weight_matrix(nrns_rec, nrns_out)

    np.save(weights_dir / "W_in_rec_initial.npy", W0_in_rec)
    np.save(weights_dir / "W_rec_rec_initial.npy", W0_rec_rec)
    np.save(weights_dir / "W_rec_out_initial.npy", W0_rec_out)

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------
    run_duration_ms = steps["total_offset"] + GROUP_SIZE * SEQUENCE_MS + steps["extension_sim"]
    log_records = []
    pred_records = []
    state = {"current_time_ms": 0.0, "global_step": 0}

    def save_partial_tables() -> None:
        pd.DataFrame(log_records).to_csv(logs_dir / "training_log_partial.csv", index=False)
        pd.DataFrame(pred_records).to_csv(logs_dir / "predictions_partial.csv", index=False)

    def run_one_group(phase: str, eta: float, loader: BalancedGroupLoader) -> None:
        set_eprop_optimizer(eta)

        current_time = state["current_time_ms"]
        group = loader.get_group()

        (
            spike_gen_params,
            target_params,
            learning_window_params,
            y_true,
            sample_ids,
            original_labels,
        ) = build_generators_for_group(group, current_time, n_classes, n_readouts, steps)

        gen_spk_in.set(spike_gen_params)
        gen_rate_target.set(target_params)
        gen_learning_window.set(learning_window_params)

        nest.Simulate(run_duration_ms)

        accuracy, loss, y_pred, scores = evaluate_readout(
            multimeter=mm_out,
            readout_nodes=nrns_out,
            current_time_ms=current_time,
            n_classes=n_classes,
            group_size=GROUP_SIZE,
            steps=steps,
            y_true=y_true,
        )

        rec_rate = recurrent_firing_rate(sr_rec, current_time, run_duration_ms, N_REC)

        log_records.append(
            {
                "global_step": state["global_step"],
                "phase": phase,
                "eta": eta,
                "accuracy": accuracy,
                "loss": loss,
                "recurrent_rate_hz": rec_rate,
                "current_time_ms": current_time,
                "run_duration_ms": run_duration_ms,
            }
        )

        for i in range(len(y_true)):
            row = {
                "global_step": state["global_step"],
                "phase": phase,
                "sample_id": sample_ids[i],
                "original_label": int(original_labels[i]),
                "true_class": int(y_true[i]),
                "pred_class": int(y_pred[i]),
                "correct": bool(y_true[i] == y_pred[i]),
            }
            for k in range(n_classes):
                row[f"score_{k}"] = float(scores[i, k])
            pred_records.append(row)

        print(
            f"[{phase:10s}] step={state['global_step']:04d} "
            f"acc={accuracy:.3f} loss={loss:.6f} rec_rate={rec_rate:.2f} Hz"
        )

        clear_recorder_buffers(mm_out, sr_rec)

        state["current_time_ms"] += run_duration_ms
        state["global_step"] += 1

        if state["global_step"] % 10 == 0:
            save_partial_tables()

    print("\n=== Training ===")
    for _ in range(N_ITER_TRAIN):
        run_one_group("train", ETA_TRAIN, train_loader)

    print("\n=== Train evaluation with eta=0 ===")
    for _ in range(N_TRAIN_EVAL_GROUPS):
        run_one_group("train_eval", ETA_EVAL, train_eval_loader)

    print("\n=== Validation with eta=0 ===")
    for _ in range(N_VAL_GROUPS):
        run_one_group("val", ETA_EVAL, val_loader)

    final_spike_time = state["current_time_ms"] + steps["offset_gen"]
    gen_spk_final_update.set({"spike_times": [final_spike_time]})
    nest.Simulate(steps["final_update"] + steps["offset_gen"])
    state["current_time_ms"] += steps["final_update"] + steps["offset_gen"]

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print("\nReading final weights...")
    W1_in_rec = get_weight_matrix(nrns_in, nrns_rec)
    W1_rec_rec = get_weight_matrix(nrns_rec, nrns_rec)
    W1_rec_out = get_weight_matrix(nrns_rec, nrns_out)

    dW_in_rec = W1_in_rec - W0_in_rec
    dW_rec_rec = W1_rec_rec - W0_rec_rec
    dW_rec_out = W1_rec_out - W0_rec_out

    np.save(weights_dir / "W_in_rec_final.npy", W1_in_rec)
    np.save(weights_dir / "W_rec_rec_final.npy", W1_rec_rec)
    np.save(weights_dir / "W_rec_out_final.npy", W1_rec_out)

    np.save(weights_dir / "W_in_rec_delta.npy", dW_in_rec)
    np.save(weights_dir / "W_rec_rec_delta.npy", dW_rec_rec)
    np.save(weights_dir / "W_rec_out_delta.npy", dW_rec_out)

    log_df = pd.DataFrame(log_records)
    pred_df = pd.DataFrame(pred_records)

    log_path = logs_dir / "training_log.csv"
    pred_path = logs_dir / "predictions.csv"
    log_df.to_csv(log_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    train_eval_pred = pred_df[pred_df["phase"] == "train_eval"]
    val_pred = pred_df[pred_df["phase"] == "val"]

    cm_train_eval = confusion_matrix_np(
        train_eval_pred["true_class"].to_numpy(),
        train_eval_pred["pred_class"].to_numpy(),
        n_classes=n_classes,
    )
    cm_val = confusion_matrix_np(
        val_pred["true_class"].to_numpy(),
        val_pred["pred_class"].to_numpy(),
        n_classes=n_classes,
    )

    train_eval_acc = float(train_eval_pred["correct"].mean())
    val_acc = float(val_pred["correct"].mean())
    train_eval_loss = float(log_df[log_df["phase"] == "train_eval"]["loss"].mean())
    val_loss = float(log_df[log_df["phase"] == "val"]["loss"].mean())
    train_eval_rate = float(log_df[log_df["phase"] == "train_eval"]["recurrent_rate_hz"].mean())
    val_rate = float(log_df[log_df["phase"] == "val"]["recurrent_rate_hz"].mean())

    pd.DataFrame(cm_train_eval).to_csv(logs_dir / "train_eval_confusion_matrix.csv", index=False)
    pd.DataFrame(cm_val).to_csv(logs_dir / "val_confusion_matrix.csv", index=False)

    config = {
        "script": "mc_maze_native_nest_eprop.py",
        "prepared_dir": str(PREPARED_DIR),
        "output_dir": str(output_dir),
        "run_name": RUN_NAME,
        "n_in": N_IN,
        "n_rec": N_REC,
        "n_classes": n_classes,
        "n_readout_segments": N_READOUT_SEGMENTS,
        "n_readouts_total": n_readouts,
        "group_size": GROUP_SIZE,
        "n_iter_train": N_ITER_TRAIN,
        "n_train_eval_groups": N_TRAIN_EVAL_GROUPS,
        "n_val_groups": N_VAL_GROUPS,
        "sequence_ms": SEQUENCE_MS,
        "eta_train": ETA_TRAIN,
        "eta_eval": ETA_EVAL,
        "target_amplitude": TARGET_AMPLITUDE,
        "background_rate_hz": BACKGROUND_RATE_HZ,
        "background_weight_native": BACKGROUND_WEIGHT,
        "wmin": WMIN,
        "wmax": WMAX,
        "keep_in_rec": KEEP_IN_REC,
        "keep_rec_rec": KEEP_REC_REC,
        "keep_rec_out": KEEP_REC_OUT,
        "W_in_scale_native": W_IN_SCALE,
        "W_rec_scale_native": W_REC_SCALE,
        "W_out_scale_native": W_OUT_SCALE,
        "feedback_scale": FEEDBACK_SCALE,
        "seed": SEED,
        "uses_nest_native_eprop": True,
        "plastic_synapse_model": "eprop_synapse",
        "recurrent_neuron_model": "eprop_iaf",
        "readout_model": "eprop_readout",
        "learning_signal_connection": "eprop_learning_signal_connection",
        "temporal_readout_native": True,
        "prepared_config": prepared_config,
    }
    save_json(config, output_dir / "config.json")

    final_metrics = {
        "train_eval_accuracy": train_eval_acc,
        "train_eval_loss": train_eval_loss,
        "train_eval_rec_rate_hz": train_eval_rate,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "val_rec_rate_hz": val_rate,
        "elapsed_seconds": elapsed,
        "n_train_samples_loaded": len(train_samples),
        "n_val_samples_loaded": len(val_samples),
        "n_in": N_IN,
        "n_rec": N_REC,
        "n_classes": n_classes,
        "n_readout_segments": N_READOUT_SEGMENTS,
        "n_readouts_total": n_readouts,
        "n_iter_train": N_ITER_TRAIN,
        "group_size": GROUP_SIZE,
        "eta_train": ETA_TRAIN,
        "target_amplitude": TARGET_AMPLITUDE,
        "sequence_ms": SEQUENCE_MS,
        "uses_nest_native_eprop": True,
        "W_in_rec_delta_norm": float(np.linalg.norm(dW_in_rec)),
        "W_rec_rec_delta_norm": float(np.linalg.norm(dW_rec_rec)),
        "W_rec_out_delta_norm": float(np.linalg.norm(dW_rec_out)),
    }
    save_json(final_metrics, output_dir / "final_metrics.json")

    print("Saving figures...")
    plot_training_curves(log_df, figures_dir)

    plot_confusion(
        cm_train_eval,
        f"Train-eval confusion matrix, acc={train_eval_acc:.2f}",
        figures_dir / "train_eval_confusion_matrix.png",
    )
    plot_confusion(
        cm_val,
        f"Validation confusion matrix, acc={val_acc:.2f}",
        figures_dir / "val_confusion_matrix.png",
    )

    for W, name in [
        (W0_in_rec, "W_in_rec_initial"),
        (W1_in_rec, "W_in_rec_final"),
        (dW_in_rec, "W_in_rec_delta"),
        (W0_rec_rec, "W_rec_rec_initial"),
        (W1_rec_rec, "W_rec_rec_final"),
        (dW_rec_rec, "W_rec_rec_delta"),
        (W0_rec_out, "W_rec_out_initial"),
        (W1_rec_out, "W_rec_out_final"),
        (dW_rec_out, "W_rec_out_delta"),
    ]:
        plot_weight_matrix(W, name.replace("_", " "), figures_dir / f"{name}.png")

    print("\n=== Final metrics ===")
    print(json.dumps(final_metrics, indent=4))
    print("\n=== Saved files ===")
    print(f"Output dir:   {output_dir}")
    print(f"Metrics:      {output_dir / 'final_metrics.json'}")
    print(f"Config:       {output_dir / 'config.json'}")
    print(f"Training log: {log_path}")
    print(f"Predictions:  {pred_path}")
    print(f"Figures dir:  {figures_dir}")
    print(f"Weights dir:  {weights_dir}")


if __name__ == "__main__":
    main()
