"""Client used by the MyST figure notebooks for the native NEST e-prop tutorial.

Primary data source
-------------------
A small Dash/Flask server exposes the training logs through a REST API.

Fallback
--------
For robust local builds, the two CSV logs are also bundled in content/data.
If the remote API is unavailable, the client transparently falls back to them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests


DEFAULT_SERVER_URL = os.getenv(
    "EPROP_SERVER_URL",
    "https://eprop-wired-dashboard.onrender.com",
)


class EpropClient:
    def __init__(self, server_url: str | None = None, timeout: int = 20):
        self.server_url = (server_url or DEFAULT_SERVER_URL).rstrip("/")
        self.api_url = f"{self.server_url}/api"
        self.timeout = timeout
        self.local_data_dir = Path(__file__).resolve().parent / "data"

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------
    def health(self) -> dict:
        response = requests.get(f"{self.api_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _local_log_path(self, dataset: str) -> Path:
        mapping = {
            "dvs_gesture": self.local_data_dir / "dvs_training_log.csv",
            "mc_maze": self.local_data_dir / "mc_maze_training_log.csv",
        }
        if dataset not in mapping:
            raise ValueError(f"Unknown dataset: {dataset}")
        return mapping[dataset]

    def get_training_log(self, dataset: str) -> pd.DataFrame:
        """Retrieve a training log from the API, with a bundled CSV fallback."""
        try:
            response = requests.get(
                f"{self.api_url}/training/{dataset}",
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            records = payload["records"] if isinstance(payload, dict) else payload
            return pd.DataFrame(records)
        except Exception:
            return pd.read_csv(self._local_log_path(dataset))

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    @staticmethod
    def _dataset_config(dataset: str) -> dict:
        configs = {
            "dvs_gesture": {
                "title": "DVS Gesture",
                "phase_order": ["train", "train_eval", "test"],
                "phase_labels": {
                    "train": "Train",
                    "train_eval": "Train-eval",
                    "test": "Test",
                },
            },
            "mc_maze": {
                "title": "MC_Maze",
                "phase_order": ["train", "train_eval", "val"],
                "phase_labels": {
                    "train": "Train",
                    "train_eval": "Train-eval",
                    "val": "Validation",
                },
            },
        }
        if dataset not in configs:
            raise ValueError(f"Unknown dataset: {dataset}")
        return configs[dataset]

    def training_figure(self, dataset: str) -> go.Figure:
        df = self.get_training_log(dataset)
        cfg = self._dataset_config(dataset)

        metrics = [
            {
                "key": "accuracy",
                "label": "Accuracy",
                "yaxis_title": "Accuracy",
                "ylim": [0.0, 1.05],
                "title_suffix": "accuracy",
            },
            {
                "key": "loss",
                "label": "MSE-like loss",
                "yaxis_title": "MSE-like loss",
                "ylim": None,
                "title_suffix": "mse-like loss",
            },
            {
                "key": "recurrent_rate_hz",
                "label": "Mean recurrent firing rate [Hz]",
                "yaxis_title": "Mean recurrent firing rate [Hz]",
                "ylim": None,
                "title_suffix": "mean recurrent firing rate [Hz]",
            },
        ]

        phase_style = {
            "train": {"color": "#1f77b4", "width": 1.6, "size": 5},
            "train_eval": {"color": "#ff7f0e", "width": 2.2, "size": 7},
            "test": {"color": "#2ca02c", "width": 2.2, "size": 7},
            "val": {"color": "#2ca02c", "width": 2.2, "size": 7},
        }

        fig = go.Figure()
        trace_indices_by_metric: dict[str, list[int]] = {}

        for metric_idx, metric_cfg in enumerate(metrics):
            trace_indices_by_metric[metric_cfg["key"]] = []

            for phase in cfg["phase_order"]:
                sub = df[df["phase"] == phase].copy().sort_values("global_step")
                if sub.empty:
                    continue

                style = phase_style[phase]
                trace_index = len(fig.data)
                trace_indices_by_metric[metric_cfg["key"]].append(trace_index)

                fig.add_trace(
                    go.Scatter(
                        x=sub["global_step"],
                        y=sub[metric_cfg["key"]],
                        mode="lines+markers",
                        name=cfg["phase_labels"][phase],
                        legendgroup=phase,
                        showlegend=True,
                        visible=(metric_idx == 0),
                        line=dict(color=style["color"], width=style["width"]),
                        marker=dict(size=style["size"], color=style["color"]),
                        hovertemplate=(
                            "Iteration: %{x}<br>"
                            + f"{metric_cfg['label']}: "
                            + "%{y:.6f}<br>"
                            + f"Phase: {cfg['phase_labels'][phase]}"
                            + "<extra></extra>"
                        ),
                    )
                )

        def visibility_for(metric_key: str) -> list[bool]:
            visible = [False] * len(fig.data)
            for index in trace_indices_by_metric[metric_key]:
                visible[index] = True
            return visible

        def yaxis_update(metric_cfg: dict) -> dict:
            update = {"title": metric_cfg["yaxis_title"]}
            if metric_cfg["ylim"] is not None:
                update["range"] = metric_cfg["ylim"]
            else:
                update["autorange"] = True
            return update

        buttons = []
        for metric_cfg in metrics:
            buttons.append(
                dict(
                    label=metric_cfg["label"],
                    method="update",
                    args=[
                        {"visible": visibility_for(metric_cfg["key"])},
                        {
                            "title": {
                                "text": (
                                    f"{cfg['title']} native NEST e-prop: "
                                    f"{metric_cfg['title_suffix']}"
                                ),
                                "x": 0.5,
                                "xanchor": "center",
                                "y": 0.97,
                                "yanchor": "top",
                            },
                            "yaxis": yaxis_update(metric_cfg),
                        },
                    ],
                )
            )

        first_metric = metrics[0]
        fig.update_layout(
            title={
                "text": (
                    f"{cfg['title']} native NEST e-prop: "
                    f"{first_metric['title_suffix']}"
                ),
                "x": 0.5,
                "xanchor": "center",
                "y": 0.97,
                "yanchor": "top",
            },
            xaxis=dict(title="Group / iteration"),
            yaxis=yaxis_update(first_metric),
            hovermode="closest",
            template="plotly_white",
            legend=dict(title="Phase"),
            margin=dict(l=70, r=30, t=170, b=70),
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=buttons,
                    x=0.0,
                    y=1.22,
                    xanchor="left",
                    yanchor="top",
                    showactive=True,
                )
            ],
            annotations=[
                dict(
                    text="Metric",
                    x=0.0,
                    y=1.285,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
        )

        return fig
