#!/usr/bin/env python3
"""
Cross-platform pipeline runner (Windows-friendly alternative to `make`).

  python run_all.py            # data -> train -> evaluate -> liberty
  python run_all.py --web      # also launch the dashboard at the end
  python run_all.py --clean    # remove generated artifacts
  python run_all.py --step data|train|evaluate|liberty|web
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args, cwd=None):
    print(f"\n>>> {' '.join(args)}")
    r = subprocess.run(args, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(args)}")


def step_data():
    run([PY, str(ROOT / "spice" / "run_sweep.py")])
    run([PY, str(ROOT / "data" / "process_data.py")])


def step_train():
    run([PY, str(ROOT / "ml" / "train.py")])


def step_evaluate():
    run([PY, str(ROOT / "ml" / "evaluate.py")])


def step_liberty():
    run([PY, str(ROOT / "data" / "gen_liberty.py")])


def step_web():
    run([PY, str(ROOT / "web" / "app.py")])


def clean():
    pats = ["data/raw/*.csv", "data/processed/*.csv", "data/processed/*.pkl",
            "data/liberty/*.lib", "ml/checkpoints/*.pt", "ml/checkpoints/*.png",
            "ml/checkpoints/*.pkl", "ml/checkpoints/*.csv"]
    n = 0
    for pat in pats:
        for f in ROOT.glob(pat):
            f.unlink()
            n += 1
    print(f"removed {n} generated files.")


STEPS = {"data": step_data, "train": step_train, "evaluate": step_evaluate,
         "liberty": step_liberty, "web": step_web}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--step", choices=list(STEPS))
    args = ap.parse_args()

    if args.clean:
        clean()
        return
    if args.step:
        STEPS[args.step]()
        return

    step_data()
    step_train()
    step_evaluate()
    step_liberty()
    if args.web:
        step_web()
    else:
        print("\nPipeline complete. Launch the dashboard with:\n"
              "    python web/app.py   (or: make web)")


if __name__ == "__main__":
    main()
