# ============================================================================
# One-command pipeline runner.
#   make all       -> data + train + evaluate
#   make data      -> run PVT sweep + process dataset
#   make train     -> train the MLP
#   make evaluate  -> benchmark ML vs SPICE
#   make liberty   -> generate the .lib file
#   make web       -> launch the Flask dashboard
#   make clean     -> remove generated artifacts
#
# Cross-platform: uses `python`. On Windows, `make clean` falls back to the
# Python cleaner (run_all.py --clean) since rm is unavailable.
# ============================================================================
PYTHON ?= python

.PHONY: all data train evaluate liberty web notebook clean

all: data train evaluate

data:
	$(PYTHON) spice/run_sweep.py
	$(PYTHON) data/process_data.py

train:
	$(PYTHON) ml/train.py

evaluate:
	$(PYTHON) ml/evaluate.py

liberty:
	$(PYTHON) data/gen_liberty.py

web:
	cd web && flask run --port 5000

notebook:
	jupyter notebook notebooks/exploration.ipynb

clean:
	$(PYTHON) run_all.py --clean
