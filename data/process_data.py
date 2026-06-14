#!/usr/bin/env python3
"""
Data processing + feature engineering.

Loads the raw per-cell sweep CSVs from data/raw/, engineers the model features,
applies log-transform to the delay targets, standard-scales the inputs, makes a
stratified 70/15/15 train/val/test split, and writes:

    data/processed/timing_dataset.csv   # full engineered dataset + split column
    data/processed/scalers.pkl          # {feature_scaler, feature_cols, ...}

Run:  python data/process_data.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

PROC_FACTOR = {"tt": 1.00, "ff": 0.85, "ss": 1.20, "fs": 1.05, "sf": 0.95}
VTH = 0.35

FEATURE_COLS = [
    "vdd", "temp", "cload_ff", "drive_strength", "proc_factor",
    "cell_type_inv", "cell_type_nand2", "cell_type_dff",
    "vdd_sq", "inv_drive", "vdd_minus_vth",
]
TARGET_COLS = ["tpd_ps", "tpHL_ps", "tpLH_ps"]


def load_raw() -> pd.DataFrame:
    frames = []
    for csv in sorted(RAW_DIR.glob("*_sweep.csv")):
        frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(
            f"No *_sweep.csv in {RAW_DIR}. Run spice/run_sweep.py first.")
    return pd.concat(frames, ignore_index=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["vdd"] = df["vdd"].astype(float)
    out["temp"] = df["temp"].astype(float)
    out["cload_ff"] = df["cload"].astype(float) * 1e15        # F -> fF
    out["drive_strength"] = df["drive"].astype(float)
    out["proc_factor"] = df["proc"].map(PROC_FACTOR).astype(float)

    out["cell_type_inv"] = (df["cell"] == "inv").astype(float)
    out["cell_type_nand2"] = (df["cell"] == "nand2").astype(float)
    out["cell_type_dff"] = (df["cell"] == "dff").astype(float)

    # nonlinear / physics-informed features
    out["vdd_sq"] = out["vdd"] ** 2
    out["inv_drive"] = 1.0 / out["drive_strength"]
    out["vdd_minus_vth"] = out["vdd"] - VTH

    # targets
    out["tpd_ps"] = df["tpd_ps"].astype(float)
    out["tpHL_ps"] = df["tpHL_ps"].astype(float)
    out["tpLH_ps"] = df["tpLH_ps"].astype(float)

    # carry through for stratification / analysis
    out["cell"] = df["cell"].values
    out["proc"] = df["proc"].values

    # log target (delay is roughly lognormal across the 10x dynamic range)
    out["tpd_ps_log"] = np.log(out["tpd_ps"].clip(lower=1e-3))
    return out


def main():
    print("[process_data] loading raw sweeps ...")
    raw = load_raw()
    df = engineer(raw)
    print(f"[process_data] {len(df)} total rows across {df['cell'].nunique()} cells")

    # ---- stratified 70/15/15 split on (cell, proc) --------------------------
    strat = df["cell"].astype(str) + "_" + df["proc"].astype(str)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=strat)
    strat_temp = temp_df["cell"].astype(str) + "_" + temp_df["proc"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=strat_temp)

    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    print(f"[process_data] split -> train={len(train_df)} "
          f"val={len(val_df)} test={len(test_df)}")

    # ---- fit StandardScaler on TRAIN features only --------------------------
    scaler = StandardScaler()
    scaler.fit(df.loc[df["split"] == "train", FEATURE_COLS].values)

    bundle = {
        "feature_scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "log_target_col": "tpd_ps_log",
        "vth": VTH,
        "proc_factor": PROC_FACTOR,
    }
    with (PROC_DIR / "scalers.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    print(f"[process_data] saved scalers -> {PROC_DIR / 'scalers.pkl'}")

    out_path = PROC_DIR / "timing_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[process_data] saved dataset -> {out_path}")


if __name__ == "__main__":
    main()
