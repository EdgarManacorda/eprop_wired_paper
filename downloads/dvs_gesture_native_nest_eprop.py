#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVS Gesture native NEST e-prop tutorial
=======================================

This script trains a recurrent spiking neural network on four DVS Gesture
classes using native NEST e-prop components.

The input event streams are converted to 16 x 16 x 2 spike channels, compressed
to a 500 ms sequence, and binned at 5 ms resolution before being sent to NEST.

Run from the project root:
    python scripts/dvs_gesture_native_nest_eprop.py

Expected data location:
    data/raw/DVS128Gesture

Main outputs:
    results/dvs_gesture/dvs_gesture_native_nest_eprop_tutorial/
        config.json
        final_metrics.json
        training_log.csv
        predictions.csv
        confusion_matrix_*.csv
        figures/*.png
        weights/*.npy
"""

from __future__ import annotations

import json
import math
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

RESULTS_ROOT = PROJECT_ROOT / "results/dvs_gesture"
RUN_NAME = "dvs_gesture_native_nest_eprop_tutorial"
SPIKINGJELLY_ROOT = PROJECT_ROOT / "data/raw/DVS128Gesture"

LABELS = [0, 1, 2, 3]
N_TRAIN_PER_LABEL = 60
N_TEST_PER_LABEL = 20

# Main parameters readers may want to change first:
#   LABELS, N_TRAIN_PER_LABEL, N_TEST_PER_LABEL, N_REC, N_ITER_TRAIN,
#   ETA_TRAIN, SEQUENCE_MS, BIN_MS, and the sparsity values below.
N_IN = 512                 # 16 x 16 x 2 DVS channels
N_REC = 150
SEQUENCE_MS = 500.0
LEARNING_WINDOW_MS = 100.0
BIN_MS = 5.0
MAX_SPIKES_PER_SAMPLE = 30000

GROUP_SIZE = 20            # 5 samples/class/group for 4 classes
N_ITER_TRAIN = 80
N_ITER_TEST = 4

ETA_TRAIN = 2e-7
ETA_EVAL = 0.0
WMIN = -3.0
WMAX = 3.0
OPTIMIZER_BATCH_SIZE = 1
OPTIMIZE_EACH_STEP = False

KEEP_IN_REC = 0.15
KEEP_REC_REC = 0.05
KEEP_REC_OUT = 1.0

W_IN_SCALE = 1.0
W_REC_SCALE = 0.8
W_OUT_SCALE = 1.0
FEEDBACK_SCALE = 0.2

TARGET_AMPLITUDE = 0.5

RESOLUTION_MS = 1.0
THREADS = 1
SEED = 18


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def node_ids(nodes) -> np.ndarray:
    if hasattr(nodes, "tolist"):
        return np.array(nodes.tolist(), dtype=int)
    return np.array(list(nodes), dtype=int)


def quiet_nest() -> None:
    try:
        nest.set_verbosity("M_FATAL")
    except Exception:
        try:
            nest.set_verbosity("M_WARNING")
        except Exception:
            pass


def filter_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only parameters supported by the installed NEST build."""
    try:
        valid = set(nest.GetDefaults(model_name).keys())
    except Exception:
        return dict(params)

    filtered = {k: v for k, v in params.items() if k in valid}
    removed = sorted(set(params) - set(filtered))
    if removed:
        print(f"[compat] {model_name}: removed unsupported parameters: {removed}")
    return filtered


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
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


def plot_training_curves(log_df: pd.DataFrame, figures_dir: Path) -> None:
    train_df = log_df[log_df["phase"] == "train"]
    test_df = log_df[log_df["phase"] == "test"]
    train_eval_df = log_df[log_df["phase"] == "train_eval"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["accuracy"], marker="o", label="train")
    ax.plot(test_df["global_step"], test_df["accuracy"], marker="o", label="test")
    ax.scatter(train_eval_df["global_step"], train_eval_df["accuracy"], marker="x", s=80, label="train_eval")
    ax.set_title("NEST native e-prop accuracy")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["loss"], marker="o", label="train")
    ax.plot(test_df["global_step"], test_df["loss"], marker="o", label="test")
    ax.scatter(train_eval_df["global_step"], train_eval_df["loss"], marker="x", s=80, label="train_eval")
    ax.set_title("NEST native e-prop loss")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("MSE-like loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_df["global_step"], train_df["recurrent_rate_hz"], marker="o", label="train recurrent rate")
    ax.plot(test_df["global_step"], test_df["recurrent_rate_hz"], marker="o", label="test recurrent rate")
    ax.set_title("Recurrent firing rate")
    ax.set_xlabel("Group / iteration")
    ax.set_ylabel("Mean recurrent firing rate [Hz]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "recurrent_firing_rate.png", dpi=200)
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


def normalize_and_bin_spikes(
    spikes_by_channel: list[np.ndarray],
    sequence_ms: float,
    bin_ms: float,
    max_spikes_per_sample: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    all_times = [np.asarray(t, dtype=float) for t in spikes_by_channel if len(t) > 0]
    all_times = [t[np.isfinite(t)] for t in all_times]
    all_times = [t for t in all_times if len(t) > 0]

    if not all_times:
        return [np.array([], dtype=float) for _ in spikes_by_channel]

    all_concat = np.concatenate(all_times)
    t_min = float(np.min(all_concat))
    t_max = float(np.max(all_concat))
    original_duration = max(1.0, t_max - t_min)

    usable_duration = max(1.0, sequence_ms - 2.0)
    scale = usable_duration / original_duration

    normalized = []
    for t in spikes_by_channel:
        t = np.asarray(t, dtype=float)
        t = t[np.isfinite(t)]
        if len(t) == 0:
            normalized.append(np.array([], dtype=float))
            continue

        t_new = (t - t_min) * scale + 1.0
        t_new = t_new[(t_new >= 1.0) & (t_new < sequence_ms)]

        if bin_ms > 0:
            t_new = np.floor(t_new / bin_ms) * bin_ms
            t_new = np.clip(t_new, 1.0, sequence_ms - 1.0)

        normalized.append(np.unique(t_new.astype(float)))

    total_spikes = sum(len(x) for x in normalized)
    if max_spikes_per_sample > 0 and total_spikes > max_spikes_per_sample:
        keep_ratio = max_spikes_per_sample / float(total_spikes)
        thinned = []
        for t in normalized:
            if len(t) == 0:
                thinned.append(t)
                continue
            k = max(1, int(np.floor(len(t) * keep_ratio)))
            if k >= len(t):
                thinned.append(t)
            else:
                idx = rng.choice(len(t), size=k, replace=False)
                thinned.append(np.sort(t[idx]))
        normalized = thinned

    return normalized


def events_to_16x16x2_spikes(
    events: Any,
    n_in: int,
    sequence_ms: float,
    bin_ms: float,
    max_spikes_per_sample: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Convert DVS128 events to 512 channels: polarity x 16 x 16."""
    if hasattr(events, "files"):
        t = np.asarray(events["t"], dtype=float)
        x = np.asarray(events["x"], dtype=int)
        y = np.asarray(events["y"], dtype=int)
        p = np.asarray(events["p"], dtype=int)
    elif isinstance(events, dict):
        t = np.asarray(events["t"], dtype=float)
        x = np.asarray(events["x"], dtype=int)
        y = np.asarray(events["y"], dtype=int)
        p = np.asarray(events["p"], dtype=int)
    elif isinstance(events, np.ndarray) and events.dtype.names is not None:
        t = np.asarray(events["t"], dtype=float)
        x = np.asarray(events["x"], dtype=int)
        y = np.asarray(events["y"], dtype=int)
        p = np.asarray(events["p"], dtype=int)
    elif isinstance(events, np.ndarray) and events.ndim == 2 and events.shape[1] >= 4:
        # Usually [t, x, y, p]; fallback [x, y, t, p].
        if np.nanmax(events[:, 1]) <= 127 and np.nanmax(events[:, 2]) <= 127:
            t = np.asarray(events[:, 0], dtype=float)
            x = np.asarray(events[:, 1], dtype=int)
            y = np.asarray(events[:, 2], dtype=int)
            p = np.asarray(events[:, 3], dtype=int)
        else:
            x = np.asarray(events[:, 0], dtype=int)
            y = np.asarray(events[:, 1], dtype=int)
            t = np.asarray(events[:, 2], dtype=float)
            p = np.asarray(events[:, 3], dtype=int)
    else:
        raise ValueError(f"Unsupported event object: {type(events)}")

    valid = (
        np.isfinite(t)
        & (x >= 0) & (x < 128)
        & (y >= 0) & (y < 128)
        & (p >= 0) & (p <= 1)
    )
    t, x, y, p = t[valid], x[valid], y[valid], p[valid]

    if len(t) == 0:
        return [np.array([], dtype=float) for _ in range(n_in)]

    if n_in != 512:
        raise ValueError("This script expects N_IN = 512 for the 16 x 16 x 2 input representation.")

    x16 = np.clip(x // 8, 0, 15)
    y16 = np.clip(y // 8, 0, 15)
    channels = p * 256 + y16 * 16 + x16

    spikes = [[] for _ in range(n_in)]
    for ch, tt in zip(channels, t):
        spikes[int(ch)].append(float(tt))

    spikes = [np.asarray(s, dtype=float) for s in spikes]
    return normalize_and_bin_spikes(spikes, sequence_ms, bin_ms, max_spikes_per_sample, rng)


def load_dvs_gesture_samples(
    root: Path,
    labels: list[int],
    label_to_class: dict[int, int],
    rng: np.random.Generator,
) -> tuple[list[Sample], list[Sample]]:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    def build_dataset(train: bool):
        try:
            return DVS128Gesture(root=str(root), train=train, data_type="event")
        except TypeError:
            return DVS128Gesture(root=str(root), train=train)

    def collect(ds, split_name: str, n_per_label: int) -> list[Sample]:
        samples = []
        count_by_label = {label: 0 for label in labels}

        for idx in rng.permutation(len(ds)):
            events, label = ds[int(idx)]
            label = int(label)

            if label not in labels or count_by_label[label] >= n_per_label:
                continue

            spikes = events_to_16x16x2_spikes(
                events=events,
                n_in=N_IN,
                sequence_ms=SEQUENCE_MS,
                bin_ms=BIN_MS,
                max_spikes_per_sample=MAX_SPIKES_PER_SAMPLE,
                rng=rng,
            )

            samples.append(
                Sample(
                    sample_id=f"{split_name}_{int(idx)}_label_{label}",
                    original_label=label,
                    class_index=label_to_class[label],
                    spikes_by_channel=spikes,
                )
            )
            count_by_label[label] += 1

            if all(count_by_label[label] >= n_per_label for label in labels):
                break

        for label in labels:
            if count_by_label[label] < n_per_label:
                raise RuntimeError(
                    f"Not enough samples for label {label} in split {split_name}: "
                    f"{count_by_label[label]} found, {n_per_label} required."
                )

        rng.shuffle(samples)
        return samples

    print("\nLoading DVS128Gesture with SpikingJelly...")
    train_samples = collect(build_dataset(train=True), "train", N_TRAIN_PER_LABEL)
    test_samples = collect(build_dataset(train=False), "test", N_TEST_PER_LABEL)
    return train_samples, test_samples


class BalancedGroupLoader:
    def __init__(self, samples: list[Sample], n_classes: int, group_size: int, rng: np.random.Generator):
        if group_size % n_classes != 0:
            raise ValueError("GROUP_SIZE must be divisible by the number of classes.")

        self.samples_by_class = {c: [] for c in range(n_classes)}
        for s in samples:
            self.samples_by_class[s.class_index].append(s)

        for c, values in self.samples_by_class.items():
            if not values:
                raise RuntimeError(f"No sample for class {c}.")

        self.n_classes = n_classes
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
    if keep_probability >= 1.0:
        return W
    if keep_probability <= 0.0:
        return np.zeros_like(W)
    return W * (rng.random(W.shape) < keep_probability)


def glorot(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3.0 * scale)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def connect_nonzero(W_post_pre: np.ndarray, pre, post, synapse_model: str, delay_ms: float) -> None:
    post_idx, pre_idx = np.nonzero(np.abs(W_post_pre) > 0.0)
    if len(pre_idx) == 0:
        return

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
    if len(conns) == 0:
        return W

    values = conns.get(["source", "target", "weight"])
    for source, target, weight in zip(values["source"], values["target"], values["weight"]):
        source = int(source)
        target = int(target)
        if source in pre_map and target in post_map:
            W[post_map[target], pre_map[source]] = float(weight)

    return W


def build_generators_for_group(
    group: list[Sample],
    current_time_ms: float,
    n_out: int,
    steps: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], np.ndarray, list[str], np.ndarray]:
    spike_times_by_input = [[] for _ in range(N_IN)]
    y_true = []
    sample_ids = []
    original_labels = []

    for gi, sample in enumerate(group):
        sample_start = current_time_ms + gi * SEQUENCE_MS
        input_offset = sample_start + steps["offset_gen"]

        y_true.append(sample.class_index)
        sample_ids.append(sample.sample_id)
        original_labels.append(sample.original_label)

        for ch, times in enumerate(sample.spikes_by_channel[:N_IN]):
            if len(times) == 0:
                continue
            abs_times = input_offset + np.asarray(times, dtype=float)
            abs_times = abs_times[abs_times > current_time_ms]
            spike_times_by_input[ch].extend(abs_times.tolist())

    spike_gen_params = [
        {"spike_times": sorted(np.unique(np.asarray(times, dtype=float)).tolist())}
        for times in spike_times_by_input
    ]

    target_params = []
    for out_idx in range(n_out):
        amp_times = []
        amp_values = []
        for gi, cls in enumerate(y_true):
            sample_start = current_time_ms + steps["total_offset"] + gi * SEQUENCE_MS
            amp_times.append(sample_start)
            amp_values.append(TARGET_AMPLITUDE if cls == out_idx else 0.0)
        target_params.append(
            {
                "amplitude_times": np.asarray(amp_times, dtype=float),
                "amplitude_values": np.asarray(amp_values, dtype=float),
            }
        )

    window_times = []
    window_values = []
    for gi in range(len(group)):
        sample_start = current_time_ms + steps["total_offset"] + gi * SEQUENCE_MS
        window_times.append(sample_start)
        window_values.append(0.0)
        window_times.append(sample_start + SEQUENCE_MS - LEARNING_WINDOW_MS)
        window_values.append(1.0)

    learning_window_params = {
        "amplitude_times": np.asarray(window_times, dtype=float),
        "amplitude_values": np.asarray(window_values, dtype=float),
    }

    return (
        spike_gen_params,
        target_params,
        learning_window_params,
        np.asarray(y_true, dtype=int),
        sample_ids,
        np.asarray(original_labels, dtype=int),
    )


def evaluate_readout(
    multimeter,
    readout_nodes,
    current_time_ms: float,
    n_out: int,
    group_size: int,
    steps: dict[str, float],
    y_true: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    out_ids = node_ids(readout_nodes)
    events = multimeter.get("events")

    if len(events["times"]) == 0:
        y_pred = np.zeros_like(y_true)
        return 0.0, float("nan"), y_pred, np.zeros((len(y_true), n_out))

    times = np.asarray(events["times"], dtype=float)
    senders = np.asarray(events["senders"], dtype=int)
    readout_signal = np.asarray(events["readout_signal"], dtype=float)

    y_pred = []
    scores_all = []
    losses = []

    for gi in range(group_size):
        win_start = current_time_ms + steps["total_offset"] + gi * SEQUENCE_MS + SEQUENCE_MS - LEARNING_WINDOW_MS
        win_end = current_time_ms + steps["total_offset"] + (gi + 1) * SEQUENCE_MS

        scores = np.zeros(n_out, dtype=float)
        for k, gid in enumerate(out_ids):
            mask = (times >= win_start) & (times < win_end) & (senders == int(gid))
            vals = readout_signal[mask]
            scores[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        target = np.zeros(n_out, dtype=float)
        target[int(y_true[gi])] = TARGET_AMPLITUDE

        y_pred.append(int(np.argmax(scores)))
        scores_all.append(scores)
        losses.append(0.5 * float(np.mean((scores - target) ** 2)))

    y_pred = np.asarray(y_pred, dtype=int)
    scores_all = np.asarray(scores_all, dtype=float)

    accuracy = float(np.mean(y_pred == y_true))
    loss = float(np.mean(losses))
    return accuracy, loss, y_pred, scores_all


def recurrent_firing_rate(spike_recorder, current_time_ms: float, run_duration_ms: float, n_rec: int) -> float:
    events = spike_recorder.get("events")
    if len(events["times"]) == 0:
        return 0.0

    times = np.asarray(events["times"], dtype=float)
    n_spikes = int(np.sum((times >= current_time_ms) & (times < current_time_ms + run_duration_ms)))
    return float(n_spikes / n_rec / max(1e-9, run_duration_ms / 1000.0))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(SEED)
    n_out = len(LABELS)
    label_to_class = {label: i for i, label in enumerate(LABELS)}

    output_dir = RESULTS_ROOT / RUN_NAME
    figures_dir = output_dir / "figures"
    weights_dir = output_dir / "weights"
    ensure_dir(figures_dir)
    ensure_dir(weights_dir)

    print("\n=== Clean NEST native e-prop on DVS Gesture ===")
    print(f"Output dir: {output_dir}")
    print(f"Labels: {LABELS}")
    print(f"N_IN={N_IN}, N_REC={N_REC}, N_OUT={n_out}")
    print(f"Samples/label: train={N_TRAIN_PER_LABEL}, test={N_TEST_PER_LABEL}")
    print(f"Sequence={SEQUENCE_MS} ms, learning window={LEARNING_WINDOW_MS} ms, bin={BIN_MS} ms")
    print(f"eta_train={ETA_TRAIN}, eta_eval={ETA_EVAL}, weight bounds=[{WMIN}, {WMAX}]")

    # ---------------------------------------------------------------------
    # Data loading and preprocessing
    # ---------------------------------------------------------------------
    train_samples, test_samples = load_dvs_gesture_samples(
        root=SPIKINGJELLY_ROOT,
        labels=LABELS,
        label_to_class=label_to_class,
        rng=rng,
    )

    print(f"\nLoaded train samples: {len(train_samples)}")
    print(f"Loaded test samples:  {len(test_samples)}")

    train_spike_counts = np.array([sum(len(x) for x in s.spikes_by_channel) for s in train_samples])
    test_spike_counts = np.array([sum(len(x) for x in s.spikes_by_channel) for s in test_samples])

    print("\nSpike count after preprocessing:")
    print(f"  train mean={train_spike_counts.mean():.1f}, median={np.median(train_spike_counts):.1f}, max={train_spike_counts.max()}")
    print(f"  test  mean={test_spike_counts.mean():.1f}, median={np.median(test_spike_counts):.1f}, max={test_spike_counts.max()}")

    train_loader = BalancedGroupLoader(train_samples, n_out, GROUP_SIZE, rng)
    test_loader = BalancedGroupLoader(test_samples, n_out, GROUP_SIZE, rng)

    # ---------------------------------------------------------------------
    # NEST setup
    # ---------------------------------------------------------------------
    nest.ResetKernel()
    nest.set(resolution=RESOLUTION_MS, print_time=False, total_num_virtual_procs=THREADS)
    quiet_nest()

    steps = {
        "sequence": SEQUENCE_MS,
        "learning_window": LEARNING_WINDOW_MS,
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

    print("\nCreating NEST network...")
    gen_spk_in = nest.Create("spike_generator", N_IN)
    nrns_in = nest.Create("parrot_neuron", N_IN)
    nrns_rec = nest.Create("eprop_iaf", N_REC, rec_params)
    nrns_out = nest.Create("eprop_readout", n_out, out_params)

    gen_rate_target = nest.Create("step_rate_generator", n_out)
    gen_learning_window = nest.Create("step_rate_generator")

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

    # ---------------------------------------------------------------------
    # Weights and connections
    # ---------------------------------------------------------------------
    print("Initializing weights...")

    W_in_rec = rng.normal(0.0, W_IN_SCALE / math.sqrt(N_IN), size=(N_REC, N_IN))
    W_in_rec = sparse_mask(W_in_rec, KEEP_IN_REC, rng)

    W_rec_rec = rng.normal(0.0, W_REC_SCALE / math.sqrt(N_REC), size=(N_REC, N_REC))
    np.fill_diagonal(W_rec_rec, 0.0)
    W_rec_rec = sparse_mask(W_rec_rec, KEEP_REC_REC, rng)
    np.fill_diagonal(W_rec_rec, 0.0)

    W_rec_out = glorot(N_REC, n_out, rng).T * W_OUT_SCALE * scale_factor
    W_rec_out = sparse_mask(W_rec_out, KEEP_REC_OUT, rng)

    W_feedback = rng.normal(0.0, FEEDBACK_SCALE, size=(N_REC, n_out))

    print(f"  W_in_rec nonzero:  {np.count_nonzero(W_in_rec)} / {W_in_rec.size}")
    print(f"  W_rec_rec nonzero: {np.count_nonzero(W_rec_rec)} / {W_rec_rec.size}")
    print(f"  W_rec_out nonzero: {np.count_nonzero(W_rec_out)} / {W_rec_out.size}")

    set_eprop_optimizer(ETA_TRAIN)

    print("Connecting network...")
    static_1ms = {"synapse_model": "static_synapse", "delay": steps["delay_in_rec"], "weight": 1.0}

    nest.Connect(gen_spk_in, nrns_in, {"rule": "one_to_one"}, static_1ms)

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
        {"rule": "all_to_all"},
        {
            "synapse_model": "rate_connection_delayed",
            "delay": steps["delay_in_rec"],
            "receptor_type": 1,
        },
    )

    nest.Connect(mm_out, nrns_out, {"rule": "all_to_all"}, static_1ms)
    nest.Connect(nrns_rec, sr_rec, {"rule": "all_to_all"}, static_1ms)

    # Used at the end to force the final optimizer update before reading weights.
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

    # ---------------------------------------------------------------------
    # Training and evaluation
    # ---------------------------------------------------------------------
    run_duration_ms = steps["total_offset"] + GROUP_SIZE * SEQUENCE_MS + steps["extension_sim"]

    log_records = []
    pred_records = []
    state = {"current_time_ms": 0.0, "global_step": 0}

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
        ) = build_generators_for_group(group, current_time, n_out, steps)

        gen_spk_in.set(spike_gen_params)
        gen_rate_target.set(target_params)
        gen_learning_window.set(learning_window_params)

        nest.Simulate(run_duration_ms)

        accuracy, loss, y_pred, scores = evaluate_readout(
            multimeter=mm_out,
            readout_nodes=nrns_out,
            current_time_ms=current_time,
            n_out=n_out,
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
            for k in range(n_out):
                row[f"score_{k}"] = float(scores[i, k])
            pred_records.append(row)

        print(
            f"[{phase:10s}] step={state['global_step']:04d} "
            f"acc={accuracy:.3f} loss={loss:.6f} rec_rate={rec_rate:.2f} Hz"
        )

        state["current_time_ms"] += run_duration_ms
        state["global_step"] += 1

    t0 = time.time()

    print("\n=== Training ===")
    for _ in range(N_ITER_TRAIN):
        run_one_group("train", ETA_TRAIN, train_loader)

    print("\n=== Train evaluation with eta=0 ===")
    run_one_group("train_eval", ETA_EVAL, train_loader)

    print("\n=== Test ===")
    for _ in range(N_ITER_TEST):
        run_one_group("test", ETA_EVAL, test_loader)

    final_spike_time = state["current_time_ms"] + steps["offset_gen"]
    gen_spk_final_update.set({"spike_times": [final_spike_time]})
    nest.Simulate(steps["final_update"] + steps["offset_gen"])
    state["current_time_ms"] += steps["final_update"] + steps["offset_gen"]

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
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

    log_path = output_dir / "training_log.csv"
    pred_path = output_dir / "predictions.csv"
    log_df.to_csv(log_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    train_eval_pred = pred_df[pred_df["phase"] == "train_eval"]
    test_pred = pred_df[pred_df["phase"] == "test"]

    cm_train_eval = confusion_matrix_np(
        train_eval_pred["true_class"].to_numpy(),
        train_eval_pred["pred_class"].to_numpy(),
        n_classes=n_out,
    )
    cm_test = confusion_matrix_np(
        test_pred["true_class"].to_numpy(),
        test_pred["pred_class"].to_numpy(),
        n_classes=n_out,
    )

    train_eval_acc = float(train_eval_pred["correct"].mean())
    test_acc = float(test_pred["correct"].mean())

    pd.DataFrame(cm_train_eval).to_csv(output_dir / "confusion_matrix_train_eval.csv", index=False)
    pd.DataFrame(cm_test).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    config = {
        "data_source": f"spikingjelly:{SPIKINGJELLY_ROOT}",
        "output_dir": str(output_dir),
        "labels": LABELS,
        "label_to_class": label_to_class,
        "n_in": N_IN,
        "n_rec": N_REC,
        "n_out": n_out,
        "sequence_ms": SEQUENCE_MS,
        "learning_window_ms": LEARNING_WINDOW_MS,
        "bin_ms": BIN_MS,
        "max_spikes_per_sample": MAX_SPIKES_PER_SAMPLE,
        "group_size": GROUP_SIZE,
        "n_iter_train": N_ITER_TRAIN,
        "n_iter_test": N_ITER_TEST,
        "eta_train": ETA_TRAIN,
        "eta_eval": ETA_EVAL,
        "wmin": WMIN,
        "wmax": WMAX,
        "target_amplitude": TARGET_AMPLITUDE,
        "seed": SEED,
        "uses_nest_native_eprop": True,
        "plastic_synapse_model": "eprop_synapse",
        "recurrent_neuron_model": "eprop_iaf",
        "readout_model": "eprop_readout",
        "learning_signal_connection": "eprop_learning_signal_connection",
        "scale_factor": scale_factor,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "train_eval_accuracy": train_eval_acc,
        "test_accuracy": test_acc,
        "W_in_rec_delta_norm": float(np.linalg.norm(dW_in_rec)),
        "W_rec_rec_delta_norm": float(np.linalg.norm(dW_rec_rec)),
        "W_rec_out_delta_norm": float(np.linalg.norm(dW_rec_out)),
    }
    save_json(config, output_dir / "config.json")

    final_metrics = {
        "train_eval_accuracy": train_eval_acc,
        "test_accuracy": test_acc,
        "elapsed_seconds": elapsed,
        "n_iter_train": N_ITER_TRAIN,
        "n_iter_test": N_ITER_TEST,
        "group_size": GROUP_SIZE,
        "labels": LABELS,
        "n_in": N_IN,
        "n_rec": N_REC,
        "n_out": n_out,
        "eta_train": ETA_TRAIN,
        "eta_eval": ETA_EVAL,
        "target_amplitude": TARGET_AMPLITUDE,
        "sequence_ms": SEQUENCE_MS,
        "learning_window_ms": LEARNING_WINDOW_MS,
        "bin_ms": BIN_MS,
        "max_spikes_per_sample": MAX_SPIKES_PER_SAMPLE,
    }
    save_json(final_metrics, output_dir / "final_metrics.json")

    print("Saving figures...")
    plot_training_curves(log_df, figures_dir)

    plot_confusion(
        cm_train_eval,
        f"Train-eval confusion matrix, acc={train_eval_acc:.2f}",
        figures_dir / "confusion_matrix_train_eval.png",
    )
    plot_confusion(
        cm_test,
        f"Test confusion matrix, acc={test_acc:.2f}",
        figures_dir / "confusion_matrix_test.png",
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
    print(f"Training log: {log_path}")
    print(f"Predictions:  {pred_path}")
    print(f"Figures dir:  {figures_dir}")
    print(f"Weights dir:  {weights_dir}")


if __name__ == "__main__":
    main()
