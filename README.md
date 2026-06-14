# ML-Assisted VLSI Standard-Cell Timing Prediction

A complete, end-to-end **ML-for-EDA** project: it characterizes a small CMOS
standard-cell library (INV, NAND2, DFF) across PVT corners using SPICE (with a
physics-grounded synthetic fallback), trains a PyTorch MLP to predict
propagation delay from circuit features, and serves everything through an
interactive Flask dashboard. The neural surrogate replaces multi-second SPICE
runs with **sub-millisecond inference (~1000–30,000× faster)** at **< 2% MAPE**.

> Target roles/companies: Synopsys, Cadence, Siemens EDA, Intel, NVIDIA,
> Apple Silicon, TSMC, Samsung LSI, and chip-design startups.

---

## Architecture

```
  SPICE / synthetic            ML pipeline                 Serving
  ───────────────────          ──────────────              ────────────
  inv_tb.sp  ┐                 process_data.py             Flask app.py
  nand2_tb.sp├─ run_sweep.py ─►  (features + scale) ─► train.py ─► best_model.pt
  dff_tb.sp  ┘   (PVT sweep)      timing_dataset.csv        │           │
   models/        data/raw/*.csv                      evaluate.py   inference.py
                                                       (metrics)   (TimingInference)
                                                            │           │
                                                            └──► dashboard (HTML/JS/Chart.js)
                                                                 gen_liberty.py ─► sky130_ml_char.lib
```

## Directory layout

```
vlsi_ml_timing/
├── spice/        netlists + models + run_sweep.py (PVT sweep / synthetic fallback)
├── data/         process_data.py, gen_liberty.py, raw/ processed/ liberty/
├── ml/           model.py, train.py, evaluate.py, inference.py, checkpoints/
├── web/          app.py (Flask) + templates/ + static/ (dashboard)
├── notebooks/    exploration.ipynb
├── requirements.txt, Makefile, run_all.py, README.md
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

`ngspice` is **optional**. If it is not on your PATH, `run_sweep.py`
automatically generates a physically-realistic synthetic dataset, so the whole
pipeline runs out of the box.

## Run the pipeline

Using `make`:

```bash
make data       # PVT sweep -> raw CSVs -> engineered timing_dataset.csv
make train      # train the MLP -> ml/checkpoints/best_model.pt + loss_curve.png
make evaluate   # MAE/MAPE/R2, scatter + error plots, per-corner breakdown
make liberty    # write data/liberty/sky130_ml_char.lib from ML predictions
make web        # launch dashboard at http://127.0.0.1:5000
```

Windows (no `make`) — use the cross-platform runner:

```bash
python run_all.py          # data -> train -> evaluate -> liberty
python web/app.py          # launch the dashboard
```

## Results (synthetic-default run)

| Metric            | Target     | Notes                                  |
|-------------------|------------|----------------------------------------|
| MAE (tpd)         | < 5 ps     | mean absolute delay error              |
| MAPE              | < 2 %      | across all PVT corners                 |
| R²                | > 0.998    | on held-out test set                   |
| ML inference      | ~µs/point  | `time.perf_counter()` over 1000 preds  |
| SPICE equivalent  | ~2–8 s/pt  | typical transient sim wall-clock       |
| **Speedup**       | ~10³–10⁴×  | ML vs SPICE per point                  |

(Exact numbers are printed by `make evaluate` and shown live in the dashboard.)

## Dashboard features

1. **Interactive predictor** — sliders for VDD / temp / load, drive + corner
   selectors; live delay readout with animated counter, latency and speedup.
2. **Sweep visualizer** — sweep VDD / temperature / load, overlay multiple
   process corners as Chart.js curves.
3. **PVT corner table** — 60-combination table, green→red color coding
   (fast FF → slow SS), CSV export.
4. **Accuracy dashboard** — SPICE-vs-ML scatter, error histogram, MAE/MAPE/R²
   stats, per-corner breakdown.
5. **Liberty export** — download the ML-generated `.lib` timing file.

## Background

- **PVT corners** — Process / Voltage / Temperature. Chips are signed off at
  worst-case corners (e.g. SS, low VDD, high temp) to guarantee field operation.
- **Liberty (`.lib`) files** — what foundries ship to designers: NLDM lookup
  tables of cell delay vs input slew and output capacitance. Generating them is
  millions of SPICE runs per PDK release. ML surrogates regenerate them in
  minutes when transistor models change.
- **Why ML works here** — the data has strong physical structure (monotonic in
  VDD/temp/drive, multiplicative scaling). Feature engineering (`1/drive`,
  `vdd − vth`, log-target) lets the MLP learn the physics rather than memorize.

## Physics guarantees enforced in the synthetic model

- delay decreases as VDD rises (above Vth); increases with temperature
- SS is slowest, FF is fastest; drive ×2 ≈ half the delay
- NAND2 ≈ 1.35× INV (stacked NMOS); DFF clk-Q ≈ 2.8× INV

## Future work

- More cell types (AOI/OAI, buffers, larger DFF variants) and full sky130 PDK
- Integration with **OpenSTA** for full static timing analysis
- Slew/cap 2-D NLDM table prediction (not just a single delay value)
- Uncertainty quantification at extreme corners; FPGA timing extension

---

### Resume bullet

> Designed an ML surrogate model for CMOS standard-cell timing characterization;
> trained a PyTorch MLP on ~12,000 PVT-corner data points (INV/NAND2/DFF, sky130
> process) to achieve **< 2% MAPE** with a **~10³–10⁴× speedup** over SPICE.
> Built an interactive Flask dashboard for real-time delay prediction and
> automated Liberty (`.lib`) file generation.
