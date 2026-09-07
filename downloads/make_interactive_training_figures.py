from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# 1. PATHS
# ============================================================
DVS_LOG = Path('/mnt/data/training_log.csv')
MC_LOG = Path('/mnt/data/training_log(1).csv')
OUT_DIR = Path('/mnt/data/interactive_training_figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. DATASET CONFIG
# ============================================================
DATASETS = {
    'dvs_gesture': {
        'title': 'DVS Gesture',
        'log_path': DVS_LOG,
        'phase_order': ['train', 'train_eval', 'test'],
        'phase_labels': {
            'train': 'Train',
            'train_eval': 'Train-eval',
            'test': 'Test',
        },
        'file_name': 'dvs_gesture_interactive_training_curves.html',
    },
    'mc_maze': {
        'title': 'MC_Maze',
        'log_path': MC_LOG,
        'phase_order': ['train', 'train_eval', 'val'],
        'phase_labels': {
            'train': 'Train',
            'train_eval': 'Train-eval',
            'val': 'Validation',
        },
        'file_name': 'mc_maze_interactive_training_curves.html',
    },
}

METRICS = [
    {
        'key': 'accuracy',
        'label': 'Accuracy',
        'yaxis_title': 'Accuracy',
        'ylim': [0.0, 1.05],
        'title_suffix': 'accuracy',
    },
    {
        'key': 'loss',
        'label': 'MSE-like loss',
        'yaxis_title': 'MSE-like loss',
        'ylim': None,
        'title_suffix': 'mse-like loss',
    },
    {
        'key': 'recurrent_rate_hz',
        'label': 'Mean recurrent firing rate [Hz]',
        'yaxis_title': 'Mean recurrent firing rate [Hz]',
        'ylim': None,
        'title_suffix': 'mean recurrent firing rate [hz]',
    },
]

PHASE_STYLE = {
    'train': {
        'color': '#1f77b4',
        'width': 1.6,
        'size': 5,
    },
    'train_eval': {
        'color': '#ff7f0e',
        'width': 2.2,
        'size': 7,
    },
    'test': {
        'color': '#2ca02c',
        'width': 2.2,
        'size': 7,
    },
    'val': {
        'color': '#2ca02c',
        'width': 2.2,
        'size': 7,
    },
}


# ============================================================
# 3. FIGURE BUILDER
# ============================================================

def make_figure(df: pd.DataFrame, cfg: dict):
    fig = go.Figure()
    trace_indices_by_metric = {}

    for metric_idx, metric_cfg in enumerate(METRICS):
        trace_indices_by_metric[metric_cfg['key']] = []

        for phase in cfg['phase_order']:
            sub = df[df['phase'] == phase].copy().sort_values('global_step')
            if sub.empty:
                continue

            style = PHASE_STYLE[phase]
            visible = (metric_idx == 0)
            trace_index = len(fig.data)
            trace_indices_by_metric[metric_cfg['key']].append(trace_index)

            fig.add_trace(
                go.Scatter(
                    x=sub['global_step'],
                    y=sub[metric_cfg['key']],
                    mode='lines+markers',
                    name=cfg['phase_labels'][phase],
                    legendgroup=phase,
                    showlegend=True,
                    visible=visible,
                    line=dict(color=style['color'], width=style['width']),
                    marker=dict(size=style['size'], color=style['color']),
                    customdata=sub[['phase']].to_numpy(),
                    hovertemplate=(
                        'Iteration: %{x}<br>'
                        + f"{metric_cfg['label']}: " + '%{y:.6f}<br>'
                        + 'Phase: ' + cfg['phase_labels'][phase]
                        + '<extra></extra>'
                    ),
                )
            )

    def visibility_for(metric_key):
        vis = [False] * len(fig.data)
        for idx in trace_indices_by_metric[metric_key]:
            vis[idx] = True
        return vis

    def yaxis_update(metric_cfg):
        update = {'title': metric_cfg['yaxis_title']}
        if metric_cfg['ylim'] is not None:
            update['range'] = metric_cfg['ylim']
        else:
            update['autorange'] = True
        return update

    buttons = []
    for metric_cfg in METRICS:
        buttons.append(
            dict(
                label=metric_cfg['label'],
                method='update',
                args=[
                    {'visible': visibility_for(metric_cfg['key'])},
                    {
                        'title': {
                            'text': f"{cfg['title']} native NEST e-prop: {metric_cfg['title_suffix']}",
                            'x': 0.5,
                            'xanchor': 'center',
                            'y': 0.97,
                            'yanchor': 'top',
                        },
                        'yaxis': yaxis_update(metric_cfg),
                    },
                ],
            )
        )

    first_metric = METRICS[0]
    fig.update_layout(
        title={
            'text': f"{cfg['title']} native NEST e-prop: {first_metric['title_suffix']}",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.97,
            'yanchor': 'top',
        },
        xaxis=dict(title='Group / iteration'),
        yaxis=yaxis_update(first_metric),
        hovermode='closest',
        template='plotly_white',
        legend=dict(title='Phase'),
        margin=dict(l=70, r=30, t=170, b=70),
        updatemenus=[
            dict(
                type='dropdown',
                direction='down',
                buttons=buttons,
                x=0.0,
                y=1.22,
                xanchor='left',
                yanchor='top',
                showactive=True,
            )
        ],
        annotations=[
            dict(
                text='Metric',
                x=0.0,
                y=1.285,
                xref='paper',
                yref='paper',
                xanchor='left',
                yanchor='top',
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    return fig


# ============================================================
# 4. EXPORT
# ============================================================

def main():
    saved = []

    for key, cfg in DATASETS.items():
        df = pd.read_csv(cfg['log_path'])
        fig = make_figure(df, cfg)
        out_path = OUT_DIR / cfg['file_name']
        fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
        saved.append(out_path)
        print(f'Saved: {out_path}')

    index_html = OUT_DIR / 'index.html'
    links = '\n'.join([
        f'<li><a href="{p.name}">{p.stem}</a></li>' for p in saved
    ])
    index_html.write_text(
        f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive training figures</title>
</head>
<body>
    <h1>Interactive training figures</h1>
    <ul>
        {links}
    </ul>
</body>
</html>''',
        encoding='utf-8'
    )
    print(f'Saved: {index_html}')


if __name__ == '__main__':
    main()
