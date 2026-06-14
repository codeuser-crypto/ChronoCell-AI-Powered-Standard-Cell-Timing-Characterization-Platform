#!/usr/bin/env python3
"""
Train the TimingPredictor MLP on the engineered dataset.

Trains on log-transformed tpd plus the two raw edge delays (multi-output).
Saves the best checkpoint, a loss curve, and prints MAE / MAPE / R2 on test.

Run:  python ml/train.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from model import TimingPredictor  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "epochs": 300,
    "batch_size": 512,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss": "HuberLoss",
    "patience": 30,
    "target": "tpd_ps_log",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bundle():
    with (PROC_DIR / "scalers.pkl").open("rb") as f:
        return pickle.load(f)


def make_targets(df, bundle):
    """3 outputs: [log(tpd), log(tpHL), log(tpLH)] -- all log-space, robust."""
    tpd = np.log(df["tpd_ps"].clip(lower=1e-3).values)
    tphl = np.log(df["tpHL_ps"].clip(lower=1e-3).values)
    tplh = np.log(df["tpLH_ps"].clip(lower=1e-3).values)
    return np.stack([tpd, tphl, tplh], axis=1).astype(np.float32)


def make_loader(df, scaler, cols, shuffle):
    X = scaler.transform(df[cols].values).astype(np.float32)
    return X


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    bundle = load_bundle()
    scaler = bundle["feature_scaler"]
    cols = bundle["feature_cols"]

    df = pd.read_csv(PROC_DIR / "timing_dataset.csv")
    splits = {s: df[df["split"] == s].reset_index(drop=True)
              for s in ("train", "val", "test")}

    Xtr = make_loader(splits["train"], scaler, cols, True)
    Xva = make_loader(splits["val"], scaler, cols, False)
    Xte = make_loader(splits["test"], scaler, cols, False)
    Ytr = make_targets(splits["train"], bundle)
    Yva = make_targets(splits["val"], bundle)
    Yte = make_targets(splits["test"], bundle)

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr))
    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                          shuffle=True, drop_last=True)
    Xva_t = torch.tensor(Xva, device=DEVICE)
    Yva_t = torch.tensor(Yva, device=DEVICE)

    model = TimingPredictor(input_dim=len(cols), output_dim=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    loss_fn = nn.HuberLoss(delta=0.1)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    hist = {"train": [], "val": []}

    print(f"[train] device={DEVICE}  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        train_loss = running / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), Yva_t).item()
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}")

        if bad_epochs >= CONFIG["patience"]:
            print(f"[train] early stopping at epoch {epoch}")
            break

    # ---- save best ----------------------------------------------------------
    model.load_state_dict(best_state)
    ckpt = {
        "model_state": best_state,
        "input_dim": len(cols),
        "output_dim": 3,
        "config": CONFIG,
    }
    torch.save(ckpt, CKPT_DIR / "best_model.pt")
    print(f"[train] saved -> {CKPT_DIR / 'best_model.pt'}")

    # ---- test metrics (back in linear ps space) -----------------------------
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.tensor(Xte, device=DEVICE)).cpu().numpy()
    pred_ps = np.exp(pred_log)          # [:,0]=tpd [:,1]=tpHL [:,2]=tpLH
    true_ps = np.exp(Yte)

    tpd_pred, tpd_true = pred_ps[:, 0], true_ps[:, 0]
    mae = np.mean(np.abs(tpd_pred - tpd_true))
    mape = np.mean(np.abs((tpd_pred - tpd_true) / tpd_true)) * 100
    ss_res = np.sum((tpd_true - tpd_pred) ** 2)
    ss_tot = np.sum((tpd_true - tpd_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\n[train] ===== TEST METRICS (tpd) =====")
    print(f"  MAE  = {mae:.3f} ps   (target < 5)")
    print(f"  MAPE = {mape:.3f} %   (target < 2)")
    print(f"  R^2  = {r2:.5f}       (target > 0.998)")

    metrics = {
        "mae_ps": float(mae), "mape_pct": float(mape), "r2": float(r2),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "epochs_ran": len(hist["train"]),
    }
    with (CKPT_DIR / "metrics.pkl").open("wb") as f:
        pickle.dump(metrics, f)

    # ---- loss curve ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(hist["train"], label="train")
        plt.plot(hist["val"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("Huber loss (log-delay)")
        plt.title("Training / validation loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(CKPT_DIR / "loss_curve.png", dpi=130)
        print(f"[train] loss curve -> {CKPT_DIR / 'loss_curve.png'}")
    except Exception as e:  # pragma: no cover
        print(f"[train] (skipped loss plot: {e})")


if __name__ == "__main__":
    main()
