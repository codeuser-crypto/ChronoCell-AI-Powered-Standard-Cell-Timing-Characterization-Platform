#!/usr/bin/env python3
"""
Fast inference wrapper around the trained TimingPredictor.

Loads the model + scalers once, then serves sub-millisecond predictions that
exactly mirror the training feature pipeline.

    from ml.inference import TimingInference
    eng = TimingInference()                       # auto-finds checkpoint
    eng.predict(cell="inv", vdd=1.8, temp=27, cload_ff=10.0, drive=2, corner="tt")
    # -> {"tpd_ps": 45.2, "tpHL_ps": 43.1, "tpLH_ps": 47.3, "latency_us": 0.12}
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from model import TimingPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = HERE / "checkpoints"

PROC_FACTOR = {"tt": 1.00, "ff": 0.85, "ss": 1.20, "fs": 1.05, "sf": 0.95}

# Rough wall-clock a real SPICE run would take per point (for the speedup badge)
SPICE_SECONDS_PER_POINT = {"inv": 2.5, "nand2": 3.5, "dff": 6.0}


class TimingInference:
    def __init__(self, ckpt_path: str | Path | None = None,
                 scalers_path: str | Path | None = None,
                 device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = Path(ckpt_path or (CKPT_DIR / "best_model.pt"))
        scalers_path = Path(scalers_path or (PROC_DIR / "scalers.pkl"))

        with scalers_path.open("rb") as f:
            self.bundle = pickle.load(f)
        self.scaler = self.bundle["feature_scaler"]
        self.feature_cols = self.bundle["feature_cols"]
        self.vth = self.bundle.get("vth", 0.35)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = TimingPredictor(
            input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Feature engineering -- MUST match data/process_data.py exactly
    # ------------------------------------------------------------------ #
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        cell = df["cell"].str.lower()
        out = pd.DataFrame(index=df.index)
        out["vdd"] = df["vdd"].astype(float)
        out["temp"] = df["temp"].astype(float)
        out["cload_ff"] = df["cload_ff"].astype(float)
        out["drive_strength"] = df["drive"].astype(float)
        out["proc_factor"] = df["corner"].str.lower().map(PROC_FACTOR).astype(float)
        out["cell_type_inv"] = (cell == "inv").astype(float)
        out["cell_type_nand2"] = (cell == "nand2").astype(float)
        out["cell_type_dff"] = (cell == "dff").astype(float)
        out["vdd_sq"] = out["vdd"] ** 2
        out["inv_drive"] = 1.0 / out["drive_strength"]
        out["vdd_minus_vth"] = out["vdd"] - self.vth
        return out[self.feature_cols]

    def _forward(self, feats: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(feats.values).astype(np.float32)
        with torch.no_grad():
            log_pred = self.model(torch.tensor(X, device=self.device)).cpu().numpy()
        return np.exp(log_pred)        # back to ps; cols = tpd, tpHL, tpLH

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, cell: str, vdd: float, temp: float, cload_ff: float,
                drive: int, corner: str) -> dict:
        df = pd.DataFrame([{
            "cell": cell, "vdd": vdd, "temp": temp,
            "cload_ff": cload_ff, "drive": drive, "corner": corner}])
        feats = self._features(df)
        t0 = time.perf_counter()
        ps = self._forward(feats)[0]
        latency_us = (time.perf_counter() - t0) * 1e6
        spice_s = SPICE_SECONDS_PER_POINT.get(cell.lower(), 3.0)
        return {
            "tpd_ps": round(float(ps[0]), 3),
            "tpHL_ps": round(float(ps[1]), 3),
            "tpLH_ps": round(float(ps[2]), 3),
            "latency_us": round(float(latency_us), 3),
            "spice_equiv_s": spice_s,
            "speedup": round(spice_s / (latency_us * 1e-6)),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized batch prediction. df needs columns:
        cell, vdd, temp, cload_ff, drive, corner."""
        feats = self._features(df.reset_index(drop=True))
        ps = self._forward(feats)
        out = df.reset_index(drop=True).copy()
        out["tpd_ps"] = ps[:, 0]
        out["tpHL_ps"] = ps[:, 1]
        out["tpLH_ps"] = ps[:, 2]
        return out

    def predict_corner_table(self, cell: str, drive: int,
                             cload_ff: float) -> pd.DataFrame:
        """Full PVT corner table (3 VDD x 4 temp x 5 proc = 60 rows)."""
        vdds = [1.62, 1.80, 1.98]
        temps = [-40, 27, 85, 125]
        corners = ["tt", "ff", "ss", "fs", "sf"]
        rows = [{"cell": cell, "vdd": v, "temp": t, "cload_ff": cload_ff,
                 "drive": drive, "corner": c}
                for v in vdds for t in temps for c in corners]
        return self.predict_batch(pd.DataFrame(rows))


if __name__ == "__main__":
    eng = TimingInference()
    print(eng.predict(cell="inv", vdd=1.8, temp=27,
                      cload_ff=10.0, drive=2, corner="tt"))
    tbl = eng.predict_corner_table("nand2", drive=4, cload_ff=10.0)
    print(tbl.head())
    print("corner-table rows:", len(tbl))
