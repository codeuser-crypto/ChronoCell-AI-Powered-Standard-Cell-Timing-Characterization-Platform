#!/usr/bin/env python3
"""
Benchmark ML predictions against the (SPICE/synthetic) ground truth.

Generates:
  * ml/checkpoints/scatter_spice_vs_ml.png   -- SPICE vs ML, colored by cell
  * ml/checkpoints/error_hist.png            -- % error distribution
  * ml/checkpoints/corner_breakdown.csv      -- accuracy per PVT corner
  * console: overall MAE/MAPE/R2 + SPICE-vs-ML speed benchmark
  * ml/checkpoints/eval_summary.pkl          -- consumed by the web dashboard

Run:  python ml/evaluate.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from inference import TimingInference, SPICE_SECONDS_PER_POINT  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = HERE / "checkpoints"


def metrics(true, pred):
    mae = np.mean(np.abs(pred - true))
    mape = np.mean(np.abs((pred - true) / true)) * 100
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, mape, r2


def main():
    df = pd.read_csv(PROC_DIR / "timing_dataset.csv")
    test = df[df["split"] == "test"].reset_index(drop=True).copy()

    eng = TimingInference()

    # build the inference frame
    infer_df = pd.DataFrame({
        "cell": test["cell"],
        "vdd": test["vdd"],
        "temp": test["temp"],
        "cload_ff": test["cload_ff"],
        "drive": test["drive_strength"],
        "corner": test["proc"],
    })
    pred = eng.predict_batch(infer_df)

    test["tpd_pred"] = pred["tpd_ps"].values
    true = test["tpd_ps"].values
    pr = test["tpd_pred"].values

    mae, mape, r2 = metrics(true, pr)
    print("===== OVERALL (tpd, test set) =====")
    print(f"  MAE  = {mae:.3f} ps")
    print(f"  MAPE = {mape:.3f} %")
    print(f"  R^2  = {r2:.5f}")

    # ---- speed benchmark ----------------------------------------------------
    N = 1000
    bench = infer_df.sample(N, replace=True, random_state=0).reset_index(drop=True)
    t0 = time.perf_counter()
    eng.predict_batch(bench)
    ml_total = time.perf_counter() - t0
    ml_per = ml_total / N
    spice_per = float(np.mean([SPICE_SECONDS_PER_POINT[c]
                               for c in infer_df["cell"].str.lower().unique()]))
    speedup = spice_per / ml_per
    print("\n===== SPEED BENCHMARK =====")
    print(f"  ML: {N} predictions in {ml_total*1e3:.2f} ms "
          f"({ml_per*1e6:.3f} us/point)")
    print(f"  SPICE: ~{spice_per:.1f} s/point")
    print(f"  Speedup: {speedup:,.0f}x")

    # ---- corner breakdown ---------------------------------------------------
    rows = []
    for (cell, proc), g in test.groupby(["cell", "proc"]):
        cmae, cmape, cr2 = metrics(g["tpd_ps"].values, g["tpd_pred"].values)
        rows.append({"cell": cell, "proc": proc, "n": len(g),
                     "mae_ps": round(cmae, 3), "mape_pct": round(cmape, 3),
                     "r2": round(cr2, 5)})
    breakdown = pd.DataFrame(rows).sort_values(["cell", "proc"])
    breakdown.to_csv(CKPT_DIR / "corner_breakdown.csv", index=False)
    print("\n===== PER-CORNER BREAKDOWN =====")
    print(breakdown.to_string(index=False))

    # ---- plots --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"inv": "#f59e0b", "nand2": "#10b981", "dff": "#60a5fa"}
        plt.figure(figsize=(6, 6))
        for cell, g in test.groupby("cell"):
            plt.scatter(g["tpd_ps"], g["tpd_pred"], s=6, alpha=0.4,
                        c=colors.get(cell, "#999"), label=cell)
        lim = [0, max(true.max(), pr.max()) * 1.05]
        plt.plot(lim, lim, "w--", lw=1)
        plt.xlabel("SPICE delay (ps)")
        plt.ylabel("ML predicted delay (ps)")
        plt.title("SPICE vs ML predicted tpd")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(CKPT_DIR / "scatter_spice_vs_ml.png", dpi=130)
        plt.close()

        err = (pr - true) / true * 100
        plt.figure(figsize=(7, 4.5))
        plt.hist(err, bins=60, color="#f59e0b", alpha=0.85)
        plt.xlabel("prediction error (%)")
        plt.ylabel("count")
        plt.title(f"Error distribution (MAPE={mape:.2f}%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(CKPT_DIR / "error_hist.png", dpi=130)
        plt.close()
        print(f"\n[evaluate] plots saved in {CKPT_DIR}")
    except Exception as e:  # pragma: no cover
        print(f"[evaluate] (skipped plots: {e})")

    # ---- summary for the dashboard ------------------------------------------
    # downsample scatter for the web payload
    samp = test.sample(min(800, len(test)), random_state=0)
    summary = {
        "mae_ps": float(mae), "mape_pct": float(mape), "r2": float(r2),
        "speedup": float(speedup), "ml_us_per_point": float(ml_per * 1e6),
        "spice_s_per_point": float(spice_per),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "scatter": [{"cell": r.cell, "spice": float(r.tpd_ps),
                     "ml": float(r.tpd_pred)} for r in samp.itertuples()],
        "error_pct": [float(x) for x in ((pr - true) / true * 100)[:2000]],
        "corner_breakdown": breakdown.to_dict(orient="records"),
    }
    with (CKPT_DIR / "eval_summary.pkl").open("wb") as f:
        pickle.dump(summary, f)
    print(f"[evaluate] dashboard summary -> {CKPT_DIR / 'eval_summary.pkl'}")


if __name__ == "__main__":
    main()
