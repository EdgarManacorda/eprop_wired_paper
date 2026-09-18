# Native NEST e-prop Wired Paper — article/client

This repository contains the MyST/NeuroLibre client for the interactive tutorial **Training Recurrent Spiking Neural Networks with Native NEST e-prop: An Interactive Tutorial — Applications to DVS Gesture and MC_Maze**.

The article follows the Wired Paper client/server architecture: the MyST article and executable figure notebooks are hosted in this repository, while the training data and interactive 3D network visualizations are served by a separate Dash/Flask application deployed on Render.

## Article preview

https://edgarmanacorda.github.io/eprop_wired_paper/

## Dashboard / API

https://eprop-wired-dashboard.onrender.com/

Dashboard source code:
https://github.com/EdgarManacorda/eprop_wired_dashboard

## Repository structure

- `paper.md` — main tutorial text in MyST Markdown
- `paper.bib` — bibliography
- `myst.yml` — MyST/NeuroLibre configuration
- `static/` — static tutorial figures and figure placeholders
- `content/figure_dvs_results.ipynb` — interactive DVS Gesture training diagnostics
- `content/figure_mc_results.ipynb` — interactive MC_Maze training diagnostics
- `content/Dash_client.py` — REST-API client and Plotly figure generation
- `downloads/` — complete commented DVS Gesture and MC_Maze native NEST e-prop training scripts
- `binder/` — Binder environment used by the interactive article
- `.github/workflows/` — GitHub Actions build and GitHub Pages deployment

## Interactive figures

The DVS Gesture and MC_Maze result figures query the deployed dashboard REST API. The metric selector switches between accuracy, MSE-like loss, and mean recurrent firing rate, while Plotly hover displays the exact values for individual points.

The learned-weight 3D network browsers are hosted by the dashboard and are linked from the article as full-page interactive visualizations.

## Local preview

```bash
pip install -r binder/requirements.txt
npm install -g mystmd
myst start
```

For a full executable build:

```bash
myst build --execute --html
```

Because the result notebooks retrieve their data from the deployed API, the dashboard must be reachable when executing a fresh build. The Render service may require a short cold-start delay after a period of inactivity.
