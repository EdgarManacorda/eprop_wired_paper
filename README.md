# Native NEST e-prop Wired Paper — article/client

This folder is the MyST/NeuroLibre client side of the Wired Paper. It was prepared from the tutorial document and the Wired Paper template provided by the supervisor.

## What is already wired

- Full tutorial text converted to `paper.md`
- Static tutorial figures 1–7 in `static/`
- DVS Gesture and MC_Maze interactive result notebooks
- `Dash_client.py` for REST-API data retrieval and Plotly rendering
- Static result placeholders
- Standalone interactive result HTMLs as downloads
- Standalone DVS/MC_Maze 3D network HTMLs as downloads
- Complete commented DVS Gesture and MC_Maze native NEST e-prop training scripts as downloads
- MyST bibliography, Binder configuration, and GitHub Pages workflow

## Before publishing

The dashboard/server is deployed at `https://eprop-wired-dashboard.onrender.com` and the article client is already configured to use it.

Next steps:

1. Create the GitHub repository `EdgarManacorda/eprop_wired_paper` and upload the contents of this folder to the repository root.
2. Replace the remaining author and affiliation placeholders in `myst.yml`.
3. Test the MyST build and the embedded interactive figures.
4. Optionally replace the first seven PNGs with higher-resolution versions later; the MyST paths can stay unchanged.

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

The client has a local CSV fallback for the two result figures, so the article can still build while the remote API is unavailable. The published Wired Paper should nevertheless point to the deployed server.
