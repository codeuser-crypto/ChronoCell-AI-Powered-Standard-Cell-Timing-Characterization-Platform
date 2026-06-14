#!/usr/bin/env python3
"""
Flask backend for the ML-Assisted VLSI Timing dashboard.

Loads the trained model ONCE at startup, then serves predictions, parameter
sweeps, PVT corner tables, dataset stats and evaluation metrics.

Run:  cd web && flask run --port 5000
  or: python web/app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ml"))
from inference import TimingInference  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "ml" / "checkpoints"
LIB_DIR = ROOT / "data" / "liberty"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ---- load once at startup --------------------------------------------------
ENGINE: TimingInference | None = None
DATASET: pd.DataFrame | None = None
EVAL_SUMMARY: dict | None = None
_CORNER_CACHE: dict = {}

CELLS = ["inv", "nand2", "dff"]
CORNERS = ["tt", "ff", "ss", "fs", "sf"]


def init_state():
    global ENGINE, DATASET, EVAL_SUMMARY
    try:
        ENGINE = TimingInference()
        print("[app] model loaded.")
    except Exception as e:
        print(f"[app] WARNING: could not load model ({e}). "
              f"Run `make train` first.")
    try:
        DATASET = pd.read_csv(PROC_DIR / "timing_dataset.csv")
    except Exception:
        DATASET = None
    try:
        with (CKPT_DIR / "eval_summary.pkl").open("rb") as f:
            EVAL_SUMMARY = pickle.load(f)
    except Exception:
        EVAL_SUMMARY = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_file(HERE / "templates" / "index.html")


@app.route("/api/cells")
def api_cells():
    return jsonify({"cells": CELLS, "corners": CORNERS,
                    "drives": [1, 2, 4, 8]})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if ENGINE is None:
        return jsonify({"error": "model not loaded; run `make train`"}), 503
    b = request.get_json(force=True)
    try:
        res = ENGINE.predict(
            cell=str(b.get("cell", "inv")).lower(),
            vdd=float(b["vdd"]), temp=float(b["temp"]),
            cload_ff=float(b["cload_ff"]), drive=int(b["drive"]),
            corner=str(b.get("corner", "tt")).lower())
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"bad request: {e}"}), 400
    return jsonify(res)


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    if ENGINE is None:
        return jsonify({"error": "model not loaded; run `make train`"}), 503
    b = request.get_json(force=True)
    cell = str(b.get("cell", "inv")).lower()
    drive = int(b.get("drive", 1))
    cload_ff = float(b.get("cload_ff", 10.0))
    corner = str(b.get("corner", "tt")).lower()
    param = b.get("sweep_param", "vdd")
    fixed = {"vdd": 1.80, "temp": 27.0, "cload_ff": cload_ff,
             "drive": drive, "corner": corner, "cell": cell}

    # default ranges per parameter
    ranges = {
        "vdd": [round(1.62 + i * (1.98 - 1.62) / 49, 4) for i in range(50)],
        "temp": [round(-40 + i * (125 + 40) / 49, 2) for i in range(50)],
        "cload_ff": [round(1 + i * (50 - 1) / 49, 3) for i in range(50)],
    }
    rng = b.get("sweep_range") or ranges.get(param, ranges["vdd"])

    rows = []
    for v in rng:
        rec = dict(fixed)
        rec[param] = v
        rows.append(rec)
    out = ENGINE.predict_batch(pd.DataFrame(rows))
    series = [{"param_value": rows[i][param],
               "tpd_ps": float(out["tpd_ps"].iloc[i]),
               "tpHL_ps": float(out["tpHL_ps"].iloc[i]),
               "tpLH_ps": float(out["tpLH_ps"].iloc[i])}
              for i in range(len(rows))]
    return jsonify({"sweep_param": param, "data": series})


@app.route("/api/corner_table")
def api_corner_table():
    if ENGINE is None:
        return jsonify({"error": "model not loaded; run `make train`"}), 503
    cell = request.args.get("cell", "inv").lower()
    drive = int(request.args.get("drive", 1))
    cload_ff = float(request.args.get("cload_ff", 10.0))
    key = (cell, drive, cload_ff)
    if key not in _CORNER_CACHE:
        tbl = ENGINE.predict_corner_table(cell, drive, cload_ff)
        _CORNER_CACHE[key] = tbl
    tbl = _CORNER_CACHE[key]
    rows = [{"vdd": float(r.vdd), "temp": float(r.temp),
             "corner": r.corner, "tpd_ps": round(float(r.tpd_ps), 3)}
            for r in tbl.itertuples()]
    return jsonify({"cell": cell, "drive": drive, "cload_ff": cload_ff,
                    "corners": CORNERS, "rows": rows})


@app.route("/api/dataset_stats")
def api_dataset_stats():
    if DATASET is None:
        return jsonify({"error": "dataset not found; run `make data`"}), 503
    df = DATASET
    by_cell = {c: int((df["cell"] == c).sum()) for c in CELLS}
    return jsonify({
        "total_rows": int(len(df)),
        "by_cell": by_cell,
        "tpd_min_ps": round(float(df["tpd_ps"].min()), 3),
        "tpd_max_ps": round(float(df["tpd_ps"].max()), 3),
        "tpd_mean_ps": round(float(df["tpd_ps"].mean()), 3),
        "splits": {s: int((df["split"] == s).sum())
                   for s in ("train", "val", "test")},
    })


@app.route("/api/model_metrics")
def api_model_metrics():
    if EVAL_SUMMARY is None:
        return jsonify({"error": "metrics not found; run `make evaluate`"}), 503
    return jsonify(EVAL_SUMMARY)


@app.route("/api/liberty")
def api_liberty():
    path = LIB_DIR / "sky130_ml_char.lib"
    if not path.exists():
        return jsonify({"error": "liberty file not found; run `make liberty`"}), 404
    return send_file(path, mimetype="text/plain",
                     as_attachment=True, download_name="sky130_ml_char.lib")


init_state()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
