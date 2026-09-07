from pathlib import Path
import json
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

WORKSPACE = Path("/workspace")

DVS_RUN_DIR = (
    WORKSPACE
    / "eprop_dvs_gesture"
    / "results"
    / "dvs_gesture"
    / "tutorial_nest_native_eprop_final_4labels_60train_20test_clean"
)

MC_RUN_DIR = (
    WORKSPACE
    / "eprop_mc_maze"
    / "results"
    / "mc_maze"
    / "tutorial_mc_maze_8classes_150rec_native_eprop_lowdrive_target07"
)

OUTPUT_ROOT = WORKSPACE / "tutorial_figures_clean_3d_unified_per_family"

DATASETS = {
    "dvs_gesture": {
        "label": "DVS Gesture",
        "run_dir": DVS_RUN_DIR,
    },
    "mc_maze": {
        "label": "MC_Maze",
        "run_dir": MC_RUN_DIR,
    },
}


# ============================================================
# 2. USER SETTINGS
# ============================================================

MAX_EDGES_STORED_PER_MATRIX = 5000

DEFAULT_TOP_K_IN_REC = 500
DEFAULT_TOP_K_REC_REC = 250
DEFAULT_TOP_K_REC_OUT = 250

MIN_TOP_K = 0
MAX_TOP_K_IN_REC = 2000
MAX_TOP_K_REC_REC = 1500
MAX_TOP_K_REC_OUT = 1500
TOP_K_STEP = 25

NODE_SIZE_INPUT = 2.8
NODE_SIZE_REC = 4.2
NODE_SIZE_OUT = 5.8

MIN_EDGE_WIDTH = 0.8
MAX_EDGE_WIDTH = 5.5

EDGE_OPACITY_MIN = 0.22
EDGE_OPACITY_MAX = 0.95


# ============================================================
# 3. PYTHON HELPERS
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_weights_dir(run_dir: Path) -> Path:
    candidates = [
        run_dir / "weights",
        run_dir / "logs" / "weights",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No weights directory found in {run_dir}")


def load_npy_if_exists(path: Path):
    if path.exists():
        return np.load(path)

    print(f"[WARNING] Missing file: {path}")
    return None


def make_recurrent_positions(n_nodes: int):
    """
    Compact 3D grid centered around zero.
    """
    side = int(np.ceil(n_nodes ** (1 / 3)))
    coords = []

    for idx in range(n_nodes):
        x = idx % side
        y = (idx // side) % side
        z = idx // (side * side)
        coords.append((x, y, z))

    coords = np.asarray(coords, dtype=float)
    coords -= coords.mean(axis=0, keepdims=True)
    coords *= 0.9

    return coords


def make_input_positions(n_nodes: int, x_position: float = -8.0):
    """
    Input layer: flat grid in the Y-Z plane.
    """
    side_y = int(np.ceil(np.sqrt(n_nodes)))
    coords = []

    for idx in range(n_nodes):
        y = idx % side_y
        z = idx // side_y
        coords.append((x_position, y, z))

    coords = np.asarray(coords, dtype=float)
    coords[:, 1] -= coords[:, 1].mean()
    coords[:, 2] -= coords[:, 2].mean()
    coords[:, 1:] *= 0.32

    return coords


def make_output_positions(n_nodes: int, x_position: float = 8.0):
    """
    Readout layer: flat grid in the Y-Z plane.
    """
    side_y = int(np.ceil(np.sqrt(n_nodes)))
    coords = []

    for idx in range(n_nodes):
        y = idx % side_y
        z = idx // side_y
        coords.append((x_position, y, z))

    coords = np.asarray(coords, dtype=float)
    coords[:, 1] -= coords[:, 1].mean()
    coords[:, 2] -= coords[:, 2].mean()
    coords[:, 1:] *= 0.75

    return coords


def select_sorted_edges(W: np.ndarray, max_edges: int):
    """
    W is stored as W[post, pre].

    Returns the strongest non-zero edges sorted by absolute value.
    """
    W = np.asarray(W, dtype=float)

    post_idx, pre_idx = np.nonzero(~np.isclose(W, 0.0))
    values = W[post_idx, pre_idx]

    if len(values) == 0:
        return {
            "pre": [],
            "post": [],
            "value": [],
        }

    order = np.argsort(np.abs(values))[::-1]
    order = order[: min(max_edges, len(order))]

    return {
        "pre": pre_idx[order].astype(int).tolist(),
        "post": post_idx[order].astype(int).tolist(),
        "value": values[order].astype(float).tolist(),
    }


def load_all_matrices(weights_dir: Path):
    matrices = {}

    for matrix_name in ["W_in_rec", "W_rec_rec", "W_rec_out"]:
        matrices[matrix_name] = {}

        initial = load_npy_if_exists(weights_dir / f"{matrix_name}_initial.npy")
        final = load_npy_if_exists(weights_dir / f"{matrix_name}_final.npy")
        delta = load_npy_if_exists(weights_dir / f"{matrix_name}_delta.npy")

        if delta is None and initial is not None and final is not None:
            delta = final - initial

        matrices[matrix_name]["initial"] = initial
        matrices[matrix_name]["final"] = final
        matrices[matrix_name]["delta"] = delta

    return matrices


def infer_sizes(matrices):
    n_input = None
    n_rec = None
    n_out = None

    # W_in_rec shape: [n_rec, n_input]
    for state in ["final", "initial", "delta"]:
        W = matrices["W_in_rec"].get(state)
        if W is not None:
            n_rec, n_input = W.shape
            break

    # W_rec_rec shape: [n_rec, n_rec]
    for state in ["final", "initial", "delta"]:
        W = matrices["W_rec_rec"].get(state)
        if W is not None:
            n_rec = W.shape[0]
            break

    # W_rec_out shape: [n_out, n_rec]
    for state in ["final", "initial", "delta"]:
        W = matrices["W_rec_out"].get(state)
        if W is not None:
            n_out = W.shape[0]
            break

    if n_input is None or n_rec is None or n_out is None:
        raise ValueError("Could not infer n_input, n_rec, or n_out from weight matrices.")

    return int(n_input), int(n_rec), int(n_out)


def matrix_description(matrix_name: str):
    if matrix_name == "W_in_rec":
        return {
            "label": "Input → recurrent",
            "pre_group": "input",
            "post_group": "recurrent",
            "pre_label": "input",
            "post_label": "recurrent",
        }

    if matrix_name == "W_rec_rec":
        return {
            "label": "Recurrent → recurrent",
            "pre_group": "recurrent",
            "post_group": "recurrent",
            "pre_label": "recurrent pre",
            "post_label": "recurrent post",
        }

    if matrix_name == "W_rec_out":
        return {
            "label": "Recurrent → readout",
            "pre_group": "recurrent",
            "post_group": "readout",
            "pre_label": "recurrent",
            "post_label": "readout",
        }

    raise ValueError(f"Unknown matrix name: {matrix_name}")


def build_payload(dataset_label: str, weights_dir: Path):
    matrices_np = load_all_matrices(weights_dir)
    n_input, n_rec, n_out = infer_sizes(matrices_np)

    positions = {
        "input": make_input_positions(n_input).tolist(),
        "recurrent": make_recurrent_positions(n_rec).tolist(),
        "readout": make_output_positions(n_out).tolist(),
    }

    matrices = {}

    for matrix_name in ["W_in_rec", "W_rec_rec", "W_rec_out"]:
        matrices[matrix_name] = {}

        for state in ["initial", "final", "delta"]:
            W = matrices_np[matrix_name].get(state)

            if W is None:
                continue

            edges = select_sorted_edges(W, MAX_EDGES_STORED_PER_MATRIX)
            values = np.asarray(edges["value"], dtype=float)

            if len(values) > 0:
                max_abs = float(np.max(np.abs(values)))
                min_value = float(np.min(values))
                max_value = float(np.max(values))
                n_pos = int(np.sum(values > 0))
                n_neg = int(np.sum(values < 0))
            else:
                max_abs = 1.0
                min_value = 0.0
                max_value = 0.0
                n_pos = 0
                n_neg = 0

            matrices[matrix_name][state] = {
                "shape": list(W.shape),
                "description": matrix_description(matrix_name),
                "edges": edges,
                "stats": {
                    "n_nonzero_total": int(np.count_nonzero(~np.isclose(W, 0.0))),
                    "n_edges_stored": int(len(edges["value"])),
                    "n_positive_stored": n_pos,
                    "n_negative_stored": n_neg,
                    "max_abs_stored": max_abs,
                    "min_value_stored": min_value,
                    "max_value_stored": max_value,
                },
            }

    payload = {
        "dataset_label": dataset_label,
        "node_positions": positions,
        "node_info": {
            "n_input": n_input,
            "n_recurrent": n_rec,
            "n_readout": n_out,
        },
        "matrices": matrices,
        "settings": {
            "default_top_k_in_rec": DEFAULT_TOP_K_IN_REC,
            "default_top_k_rec_rec": DEFAULT_TOP_K_REC_REC,
            "default_top_k_rec_out": DEFAULT_TOP_K_REC_OUT,
            "min_top_k": MIN_TOP_K,
            "max_top_k_in_rec": MAX_TOP_K_IN_REC,
            "max_top_k_rec_rec": MAX_TOP_K_REC_REC,
            "max_top_k_rec_out": MAX_TOP_K_REC_OUT,
            "top_k_step": TOP_K_STEP,
            "node_size_input": NODE_SIZE_INPUT,
            "node_size_rec": NODE_SIZE_REC,
            "node_size_out": NODE_SIZE_OUT,
            "min_edge_width": MIN_EDGE_WIDTH,
            "max_edge_width": MAX_EDGE_WIDTH,
            "edge_opacity_min": EDGE_OPACITY_MIN,
            "edge_opacity_max": EDGE_OPACITY_MAX,
        },
    }

    return payload


# ============================================================
# 4. HTML TEMPLATE
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>__DATASET_LABEL__ unified 3D e-prop browser</title>
    <script src="https://cdn.plot.ly/plotly-3.7.0.min.js"></script>

    <style>
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #ffffff;
            color: #222222;
        }

        .page {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .header {
            padding: 14px 22px 10px 22px;
            border-bottom: 1px solid #dddddd;
            background: #f8f8f8;
        }

        .header h1 {
            margin: 0;
            font-size: 23px;
            font-weight: 600;
        }

        .header p {
            margin: 6px 0 0 0;
            color: #555555;
            font-size: 14px;
        }

        .controls {
            display: grid;
            grid-template-columns: 145px 180px 1fr 130px;
            gap: 14px;
            align-items: center;
            padding: 12px 22px;
            border-bottom: 1px solid #dddddd;
            background: #ffffff;
            font-size: 13px;
        }

        .control-block {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .checkboxes {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .checkboxes label {
            font-weight: 500;
        }

        .sliders {
            display: grid;
            grid-template-columns: 1fr;
            gap: 6px;
        }

        .slider-block {
            display: grid;
            grid-template-columns: 120px 1fr 45px;
            gap: 8px;
            align-items: center;
        }

        label {
            font-weight: 600;
        }

        select {
            font-size: 13px;
            padding: 5px 8px;
        }

        input[type="range"] {
            width: 100%;
        }

        button {
            font-size: 13px;
            padding: 6px 10px;
            border: 1px solid #bbbbbb;
            background: #f4f4f4;
            border-radius: 4px;
            cursor: pointer;
        }

        button:hover {
            background: #e8e8e8;
        }

        .main {
            display: flex;
            flex: 1;
            min-height: 0;
        }

        #plot {
            flex: 1;
            min-width: 0;
        }

        .sidebar {
            width: 330px;
            padding: 16px;
            border-left: 1px solid #dddddd;
            background: #fafafa;
            font-size: 13px;
            overflow-y: auto;
        }

        .sidebar h2 {
            font-size: 16px;
            margin: 0 0 10px 0;
        }

        .stat {
            margin: 8px 0;
        }

        .section-title {
            margin-top: 16px;
            font-weight: 700;
        }

        .legend-box {
            margin-top: 18px;
        }

        .gradient {
            width: 245px;
            height: 20px;
            border: 1px solid #999999;
            background: linear-gradient(to right, #1b4edb, #f2f2f2, #d7191c);
            margin-top: 8px;
        }

        .gradient-labels {
            display: flex;
            justify-content: space-between;
            width: 245px;
            font-size: 11px;
            margin-top: 4px;
            color: #444444;
        }

        .note {
            font-size: 12px;
            color: #555555;
            margin-top: 14px;
            line-height: 1.35;
        }

        code {
            background: #eeeeee;
            padding: 1px 4px;
            border-radius: 3px;
        }
    </style>
</head>

<body>
<div class="page">

    <div class="header">
        <h1>__DATASET_LABEL__: unified interactive 3D e-prop network browser</h1>
        <p>
            Display input→recurrent, recurrent→recurrent, and recurrent→readout connections
            separately or together. Each connection family has its own top-k slider.
        </p>
    </div>

    <div class="controls">

        <div class="control-block">
            <label for="stateSelect">State</label>
            <select id="stateSelect">
                <option value="initial">initial</option>
                <option value="final">final</option>
                <option value="delta" selected>delta</option>
            </select>
        </div>

        <div class="checkboxes">
            <label><input type="checkbox" id="showInRec"> W_in_rec</label>
            <label><input type="checkbox" id="showRecRec" checked> W_rec_rec</label>
            <label><input type="checkbox" id="showRecOut"> W_rec_out</label>
        </div>

        <div class="sliders">
            <div class="slider-block">
                <label for="topKInRec">W_in_rec</label>
                <input
                    type="range"
                    id="topKInRec"
                    min="__MIN_TOP_K__"
                    max="__MAX_TOP_K_IN_REC__"
                    step="__TOP_K_STEP__"
                    value="__DEFAULT_TOP_K_IN_REC__"
                >
                <span id="topKInRecValue">__DEFAULT_TOP_K_IN_REC__</span>
            </div>

            <div class="slider-block">
                <label for="topKRecRec">W_rec_rec</label>
                <input
                    type="range"
                    id="topKRecRec"
                    min="__MIN_TOP_K__"
                    max="__MAX_TOP_K_REC_REC__"
                    step="__TOP_K_STEP__"
                    value="__DEFAULT_TOP_K_REC_REC__"
                >
                <span id="topKRecRecValue">__DEFAULT_TOP_K_REC_REC__</span>
            </div>

            <div class="slider-block">
                <label for="topKRecOut">W_rec_out</label>
                <input
                    type="range"
                    id="topKRecOut"
                    min="__MIN_TOP_K__"
                    max="__MAX_TOP_K_REC_OUT__"
                    step="__TOP_K_STEP__"
                    value="__DEFAULT_TOP_K_REC_OUT__"
                >
                <span id="topKRecOutValue">__DEFAULT_TOP_K_REC_OUT__</span>
            </div>
        </div>

        <button id="resetCameraButton">Reset camera</button>
    </div>

    <div class="main">
        <div id="plot"></div>

        <div class="sidebar">
            <h2>Current view</h2>

            <div class="stat"><b>Dataset:</b> <span id="datasetText"></span></div>
            <div class="stat"><b>State:</b> <span id="stateText"></span></div>
            <div class="stat"><b>Selected families:</b> <span id="familiesText"></span></div>

            <div class="section-title">Network size</div>
            <div class="stat"><b>Input neurons:</b> <span id="nInputText"></span></div>
            <div class="stat"><b>Recurrent neurons:</b> <span id="nRecText"></span></div>
            <div class="stat"><b>Readout neurons:</b> <span id="nOutText"></span></div>

            <div class="section-title">Displayed edges</div>
            <div class="stat"><b>W_in_rec:</b> <span id="edgesInRecText"></span></div>
            <div class="stat"><b>W_rec_rec:</b> <span id="edgesRecRecText"></span></div>
            <div class="stat"><b>W_rec_out:</b> <span id="edgesRecOutText"></span></div>
            <div class="stat"><b>Total displayed:</b> <span id="edgesTotalText"></span></div>

            <div class="section-title">Total non-zero edges</div>
            <div class="stat"><b>W_in_rec:</b> <span id="totalInRecText"></span></div>
            <div class="stat"><b>W_rec_rec:</b> <span id="totalRecRecText"></span></div>
            <div class="stat"><b>W_rec_out:</b> <span id="totalRecOutText"></span></div>

            <div class="legend-box">
                <b>Color scale</b>
                <div class="gradient"></div>
                <div class="gradient-labels">
                    <span id="leftColorLabel">decreased</span>
                    <span>0</span>
                    <span id="rightColorLabel">increased</span>
                </div>
            </div>

            <div class="note">
                For <code>delta</code>, colors encode weight change:
                blue means the weight decreased, red means it increased.
                For <code>initial</code> and <code>final</code>, colors encode signed weight value.
            </div>

            <div class="note">
                Each connection family has its own top-k slider. This prevents large input or recurrent weights
                from hiding weaker readout connections.
            </div>

            <div class="note">
                Edge thickness is proportional to the absolute value of the selected weight or weight change.
                Use fewer edges for a clearer figure.
            </div>
        </div>
    </div>

</div>

<script>
const PAYLOAD = __PAYLOAD_JSON__;

let currentCamera = null;

function getPositions(groupName) {
    return PAYLOAD.node_positions[groupName];
}

function rgbColor(value, maxAbs) {
    if (maxAbs <= 0) {
        return "rgba(160,160,160,0.35)";
    }

    const normalized = Math.max(-1, Math.min(1, value / maxAbs));
    const absn = Math.abs(normalized);

    const alpha = PAYLOAD.settings.edge_opacity_min +
        absn * (PAYLOAD.settings.edge_opacity_max - PAYLOAD.settings.edge_opacity_min);

    if (normalized > 0) {
        return `rgba(215,25,28,${alpha.toFixed(3)})`;
    }

    if (normalized < 0) {
        return `rgba(27,78,219,${alpha.toFixed(3)})`;
    }

    return "rgba(190,190,190,0.25)";
}

function edgeWidth(value, maxAbs) {
    if (maxAbs <= 0) {
        return PAYLOAD.settings.min_edge_width;
    }

    const normalized = Math.min(1, Math.abs(value) / maxAbs);
    return PAYLOAD.settings.min_edge_width +
        normalized * (PAYLOAD.settings.max_edge_width - PAYLOAD.settings.min_edge_width);
}

function makeNodeTrace(groupName, name, color, size) {
    const pos = getPositions(groupName);

    if (!pos || pos.length === 0) {
        return null;
    }

    return {
        type: "scatter3d",
        mode: "markers",
        x: pos.map(p => p[0]),
        y: pos.map(p => p[1]),
        z: pos.map(p => p[2]),
        marker: {
            size: size,
            color: color,
            line: {
                color: "rgba(0,0,0,0.35)",
                width: 0.5
            }
        },
        text: pos.map((p, i) => `${name} neuron ${i}`),
        hoverinfo: "text",
        name: name,
        showlegend: true
    };
}

function selectedFamilies() {
    const families = [];

    if (document.getElementById("showInRec").checked) {
        families.push("W_in_rec");
    }

    if (document.getElementById("showRecRec").checked) {
        families.push("W_rec_rec");
    }

    if (document.getElementById("showRecOut").checked) {
        families.push("W_rec_out");
    }

    return families;
}

function familyName(matrixName) {
    if (matrixName === "W_in_rec") return "input→recurrent";
    if (matrixName === "W_rec_rec") return "recurrent→recurrent";
    if (matrixName === "W_rec_out") return "recurrent→readout";
    return matrixName;
}

function getTopKForFamily(family) {
    if (family === "W_in_rec") {
        return parseInt(document.getElementById("topKInRec").value);
    }

    if (family === "W_rec_rec") {
        return parseInt(document.getElementById("topKRecRec").value);
    }

    if (family === "W_rec_out") {
        return parseInt(document.getElementById("topKRecOut").value);
    }

    return 0;
}

function getNodeGroupsForFamilies(families) {
    const groups = new Set();

    for (const family of families) {
        const matrixData =
            PAYLOAD.matrices[family]["initial"]
            || PAYLOAD.matrices[family]["final"]
            || PAYLOAD.matrices[family]["delta"];

        if (!matrixData) continue;

        groups.add(matrixData.description.pre_group);
        groups.add(matrixData.description.post_group);
    }

    return Array.from(groups);
}

function collectEdges(families, stateName) {
    let allEdges = [];

    const counts = {
        "W_in_rec": {displayed: 0, total: 0},
        "W_rec_rec": {displayed: 0, total: 0},
        "W_rec_out": {displayed: 0, total: 0},
    };

    for (const family of families) {
        const matrixData = PAYLOAD.matrices[family][stateName];

        if (!matrixData) continue;

        const edges = matrixData.edges;
        const topK = getTopKForFamily(family);
        const nEdges = Math.min(topK, edges.value.length);

        counts[family].total = matrixData.stats.n_nonzero_total;
        counts[family].displayed = nEdges;

        for (let i = 0; i < nEdges; i++) {
            allEdges.push({
                family: family,
                pre: edges.pre[i],
                post: edges.post[i],
                value: edges.value[i],
                absValue: Math.abs(edges.value[i]),
            });
        }
    }

    return {
        edges: allEdges,
        counts: counts,
    };
}

function makeEdgeTraces(edgeObjects, maxAbs) {
    const traces = [];

    for (const edge of edgeObjects) {
        const matrixData =
            PAYLOAD.matrices[edge.family]["initial"]
            || PAYLOAD.matrices[edge.family]["final"]
            || PAYLOAD.matrices[edge.family]["delta"];

        const desc = matrixData.description;

        const prePositions = getPositions(desc.pre_group);
        const postPositions = getPositions(desc.post_group);

        const p0 = prePositions[edge.pre];
        const p1 = postPositions[edge.post];

        if (!p0 || !p1) continue;

        traces.push({
            type: "scatter3d",
            mode: "lines",
            x: [p0[0], p1[0], null],
            y: [p0[1], p1[1], null],
            z: [p0[2], p1[2], null],
            line: {
                color: rgbColor(edge.value, maxAbs),
                width: edgeWidth(edge.value, maxAbs)
            },
            text:
                `matrix: ${edge.family}<br>` +
                `family: ${familyName(edge.family)}<br>` +
                `pre ${desc.pre_label}: ${edge.pre}<br>` +
                `post ${desc.post_label}: ${edge.post}<br>` +
                `value: ${edge.value.toFixed(6)}`,
            hoverinfo: "text",
            showlegend: false
        });
    }

    return traces;
}

function makeColorbarTrace(titleText, maxAbs) {
    return {
        type: "scatter3d",
        mode: "markers",
        x: [0, 0],
        y: [0, 0],
        z: [0, 0],
        marker: {
            size: 0.1,
            opacity: 0.0,
            color: [-maxAbs, maxAbs],
            colorscale: [
                [0.0, "#1b4edb"],
                [0.5, "#f2f2f2"],
                [1.0, "#d7191c"]
            ],
            cmin: -maxAbs,
            cmax: maxAbs,
            showscale: true,
            colorbar: {
                title: titleText,
                thickness: 18,
                len: 0.65,
                x: 0.98
            }
        },
        hoverinfo: "skip",
        showlegend: false
    };
}

function updateSliderLabels() {
    document.getElementById("topKInRecValue").innerText =
        document.getElementById("topKInRec").value;

    document.getElementById("topKRecRecValue").innerText =
        document.getElementById("topKRecRec").value;

    document.getElementById("topKRecOutValue").innerText =
        document.getElementById("topKRecOut").value;
}

function buildFigure() {
    const stateName = document.getElementById("stateSelect").value;
    const families = selectedFamilies();

    if (families.length === 0) {
        alert("Please select at least one connection family.");
        document.getElementById("showRecRec").checked = true;
        return buildFigure();
    }

    updateSliderLabels();

    const collected = collectEdges(families, stateName);
    const edges = collected.edges;
    const counts = collected.counts;

    let maxAbs = 1.0;

    if (edges.length > 0) {
        maxAbs = Math.max(...edges.map(e => Math.abs(e.value)));
    }

    let traces = [];

    traces.push(...makeEdgeTraces(edges, maxAbs));

    const nodeGroups = getNodeGroupsForFamilies(families);

    if (nodeGroups.includes("input")) {
        traces.push(
            makeNodeTrace(
                "input",
                "input",
                "rgba(35,170,120,0.78)",
                PAYLOAD.settings.node_size_input
            )
        );
    }

    if (nodeGroups.includes("recurrent")) {
        traces.push(
            makeNodeTrace(
                "recurrent",
                "recurrent",
                "rgba(40,100,220,0.88)",
                PAYLOAD.settings.node_size_rec
            )
        );
    }

    if (nodeGroups.includes("readout")) {
        traces.push(
            makeNodeTrace(
                "readout",
                "readout",
                "rgba(225,80,60,0.92)",
                PAYLOAD.settings.node_size_out
            )
        );
    }

    traces = traces.filter(t => t !== null);

    const colorbarTitle = stateName === "delta" ? "Δ weight" : "weight";
    traces.push(makeColorbarTrace(colorbarTitle, maxAbs));

    const selectedLabel = families.map(familyName).join(" + ");

    const stateDescription = stateName === "delta"
        ? "weight change after learning"
        : `${stateName} signed weight values`;

    const title =
        `${PAYLOAD.dataset_label}: ${selectedLabel} (${stateName})` +
        `<br><sup>color = ${stateDescription}; top-k controlled separately for each connection family</sup>`;

    const layout = {
        title: {
            text: title,
            font: {size: 20}
        },
        showlegend: true,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        margin: {l: 0, r: 0, t: 80, b: 0},
        scene: {
            xaxis: {
                title: "",
                showbackground: false,
                showticklabels: false,
                zeroline: false
            },
            yaxis: {
                title: "",
                showbackground: false,
                showticklabels: false,
                zeroline: false
            },
            zaxis: {
                title: "",
                showbackground: false,
                showticklabels: false,
                zeroline: false
            },
            aspectmode: "data",
            camera: currentCamera || {
                eye: {x: 1.8, y: 1.6, z: 1.2}
            }
        }
    };

    const config = {
        responsive: true,
        displaylogo: false
    };

    Plotly.react("plot", traces, layout, config).then(() => {
        const plotDiv = document.getElementById("plot");

        plotDiv.on("plotly_relayout", function(eventData) {
            if (eventData["scene.camera"]) {
                currentCamera = eventData["scene.camera"];
            }
        });
    });

    document.getElementById("datasetText").innerText = PAYLOAD.dataset_label;
    document.getElementById("stateText").innerText = stateName;
    document.getElementById("familiesText").innerText = families.join(", ");

    document.getElementById("nInputText").innerText = PAYLOAD.node_info.n_input;
    document.getElementById("nRecText").innerText = PAYLOAD.node_info.n_recurrent;
    document.getElementById("nOutText").innerText = PAYLOAD.node_info.n_readout;

    document.getElementById("edgesInRecText").innerText = counts["W_in_rec"].displayed;
    document.getElementById("edgesRecRecText").innerText = counts["W_rec_rec"].displayed;
    document.getElementById("edgesRecOutText").innerText = counts["W_rec_out"].displayed;
    document.getElementById("edgesTotalText").innerText = edges.length;

    document.getElementById("totalInRecText").innerText = counts["W_in_rec"].total;
    document.getElementById("totalRecRecText").innerText = counts["W_rec_rec"].total;
    document.getElementById("totalRecOutText").innerText = counts["W_rec_out"].total;

    if (stateName === "delta") {
        document.getElementById("leftColorLabel").innerText = "decreased";
        document.getElementById("rightColorLabel").innerText = "increased";
    } else {
        document.getElementById("leftColorLabel").innerText = "negative";
        document.getElementById("rightColorLabel").innerText = "positive";
    }
}

document.getElementById("stateSelect").addEventListener("change", buildFigure);

document.getElementById("showInRec").addEventListener("change", buildFigure);
document.getElementById("showRecRec").addEventListener("change", buildFigure);
document.getElementById("showRecOut").addEventListener("change", buildFigure);

document.getElementById("topKInRec").addEventListener("input", buildFigure);
document.getElementById("topKRecRec").addEventListener("input", buildFigure);
document.getElementById("topKRecOut").addEventListener("input", buildFigure);

document.getElementById("resetCameraButton").addEventListener("click", function() {
    currentCamera = null;
    buildFigure();
});

buildFigure();
</script>

</body>
</html>
"""


# ============================================================
# 5. WRITE HTML
# ============================================================

def write_html(payload: dict, output_path: Path):
    html = HTML_TEMPLATE

    html = html.replace("__DATASET_LABEL__", payload["dataset_label"])
    html = html.replace("__PAYLOAD_JSON__", json.dumps(payload))

    html = html.replace("__DEFAULT_TOP_K_IN_REC__", str(payload["settings"]["default_top_k_in_rec"]))
    html = html.replace("__DEFAULT_TOP_K_REC_REC__", str(payload["settings"]["default_top_k_rec_rec"]))
    html = html.replace("__DEFAULT_TOP_K_REC_OUT__", str(payload["settings"]["default_top_k_rec_out"]))

    html = html.replace("__MIN_TOP_K__", str(payload["settings"]["min_top_k"]))
    html = html.replace("__MAX_TOP_K_IN_REC__", str(payload["settings"]["max_top_k_in_rec"]))
    html = html.replace("__MAX_TOP_K_REC_REC__", str(payload["settings"]["max_top_k_rec_rec"]))
    html = html.replace("__MAX_TOP_K_REC_OUT__", str(payload["settings"]["max_top_k_rec_out"]))
    html = html.replace("__TOP_K_STEP__", str(payload["settings"]["top_k_step"]))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def write_index(output_root: Path):
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Unified 3D e-prop network browsers</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 40px;
            line-height: 1.5;
        }
        a {
            display: block;
            margin: 14px 0;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <h1>Unified interactive 3D e-prop network browsers</h1>
    <p>
        Select a dataset below. Each browser can display W_in_rec, W_rec_rec,
        W_rec_out, or any combination of them. Each connection family has its own top-k slider.
    </p>

    <a href="dvs_gesture/unified_3d_network_browser_per_family.html">DVS Gesture unified 3D browser</a>
    <a href="mc_maze/unified_3d_network_browser_per_family.html">MC_Maze unified 3D browser</a>
</body>
</html>
"""
    with open(output_root / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================
# 6. MAIN
# ============================================================

def process_dataset(dataset_key: str, cfg: dict):
    dataset_label = cfg["label"]
    run_dir = cfg["run_dir"]

    print("")
    print("=" * 80)
    print(f"Processing {dataset_label}")
    print("=" * 80)

    weights_dir = find_weights_dir(run_dir)

    print(f"Run dir:     {run_dir}")
    print(f"Weights dir: {weights_dir}")

    payload = build_payload(
        dataset_label=dataset_label,
        weights_dir=weights_dir,
    )

    out_dir = OUTPUT_ROOT / dataset_key
    ensure_dir(out_dir)

    output_html = out_dir / "unified_3d_network_browser_per_family.html"
    write_html(payload, output_html)

    metadata = {
        "dataset_label": dataset_label,
        "run_dir": str(run_dir),
        "weights_dir": str(weights_dir),
        "output_html": str(output_html),
        "description": (
            "Unified 3D e-prop network browser with separate top-k sliders "
            "for W_in_rec, W_rec_rec and W_rec_out."
        ),
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved HTML: {output_html}")


def main():
    ensure_dir(OUTPUT_ROOT)

    for dataset_key, cfg in DATASETS.items():
        process_dataset(dataset_key, cfg)

    write_index(OUTPUT_ROOT)

    print("")
    print("All done.")
    print(f"Open: {OUTPUT_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()