# PROJECT GUIDE — Explained

## PART 1 — THE BIG PICTURE

### The one-sentence version
> We taught a small AI to instantly predict how fast a digital logic gate is,
> replacing a slow physics simulation that used to take seconds — and wrapped it
> in a website where you can play with it.

### The 30-second version
Every computer chip is built from millions of tiny switches called
**transistors**, grouped into small reusable building blocks called
**standard cells** (think LEGO bricks: an inverter, a NAND gate, a flip-flop).

Before chip designers can use these bricks, someone has to measure **how long
each brick takes to react** — its *delay*, measured in **picoseconds**
(trillionths of a second). This measurement is done with a slow, accurate
physics simulator called **SPICE**. One measurement can take several seconds,
and a real chip needs *millions* of them. That's weeks of computer time.

**My idea:** instead of running the slow simulator every time, we run it (or a
realistic stand-in) enough times to collect example data, then train a small
**neural network** to learn the pattern. Afterwards, the AI predicts the delay
in *microseconds* (millionths of a second) — roughly **half a million times
faster** — with less than **2% error**.

Then we built a **website dashboard** so you can move sliders and see delays
update live, run experiments, and even export the results in the exact file
format that real chip foundries use.

### Why anyone would care (the "so what")
- Chip companies (Intel, NVIDIA, Apple, Synopsys, Cadence...) spend enormous
  computer time on this "characterization" step.
- A fast, accurate AI shortcut saves time and money — this is a hot research
  area called **ML for EDA** (Machine Learning for Electronic Design
  Automation).
- This project is a *miniature, working demonstration* of that idea, end to end.

---

## PART 2 — THE KEY WORDS

| Term | Plain meaning |
|---|---|
| **Transistor** | A microscopic electrical on/off switch. The atom of all chips. |
| **Standard cell** | A small pre-built logic block made of a few transistors (e.g. an inverter). Chips are assembled from thousands of copies. |
| **INV (inverter)** | The simplest gate: input 0 → output 1, input 1 → output 0. |
| **NAND2** | A 2-input gate; slightly more complex/slower than an inverter. |
| **DFF (D flip-flop)** | A 1-bit memory element that stores a value on each clock tick. The slowest of our three cells. |
| **Propagation delay (tpd)** | How long the output takes to react after the input changes. Measured in **picoseconds (ps)**. This is the number we predict. |
| **tpHL / tpLH** | Delay when output goes High→Low / Low→High. We predict both; tpd is their average. |
| **SPICE** | The accurate-but-slow circuit physics simulator. The "ground truth." |
| **ngspice** | A specific free version of SPICE. Optional here. |
| **PVT corners** | **P**rocess, **V**oltage, **T**emperature — the conditions a chip must survive. We test every combination (more below). |
| **Liberty file (.lib)** | The standard text file format foundries ship containing all these delay numbers. We generate one. |
| **Neural network / MLP** | The AI model. "MLP" = Multi-Layer Perceptron, the simplest useful kind. |
| **Feature** | An input number we feed the AI (voltage, temperature, etc.). |
| **Label / target** | The answer we want the AI to predict (the delay). |
| **Inference** | Using a trained model to make a prediction (the fast part). |
| **Flask** | The Python tool we used to build the website's backend. |

### What is a "PVT corner"? (this impresses interviewers)
A chip must work in many different real-world conditions. We can't just test the
"normal" case — we have to test the *worst* cases too. The three axes:

- **P = Process**: manufacturing isn't perfect. Some chips come out slightly
  "fast" (FF), some "slow" (SS), some typical (TT), and mixed (FS, SF).
- **V = Voltage**: the power supply can sag or spike (we use 1.62 V, 1.80 V,
  1.98 V — i.e. nominal ±10%).
- **T = Temperature**: a chip in your pocket vs. in a hot server (−40 °C up to
  125 °C).

A **corner** is one specific combination, e.g. "slow process, low voltage, hot."
Engineers must guarantee the chip still works at the *worst* corner — that's
called **sign-off**. We sweep all combinations to build our dataset.

---

## PART 3 — THE BASIC LOGIC

Think of it as an assembly line with 6 stations:

```
 [1] GENERATE DATA  ->  [2] CLEAN/PREP  ->  [3] TRAIN AI  ->  [4] TEST AI
                                                                  |
                          [6] WEBSITE  <-  [5] USE AI (predict) <-+
```

### Station 1 — Generate the example data
*File: `spice/run_sweep.py`*

We need thousands of examples of "these conditions → this delay." Two ways:

1. **The real way:** run the SPICE simulator on each PVT corner. Accurate but
   slow, and needs ngspice installed.
2. **The fallback way (what runs by default):** a **physics formula** that
   produces realistic delays instantly. It's based on a real textbook equation
   (the *alpha-power law*) and obeys the true physics rules:
   - higher voltage → faster (smaller delay)
   - higher temperature → slower
   - "slow" process → slowest, "fast" process → fastest
   - double the transistor size (drive strength) → about half the delay
   - NAND2 ≈ 1.35× slower than INV; DFF ≈ 2.8× slower than INV

> **Why a fallback?** SPICE/ngspice is awkward to install, especially on
> Windows. The fallback means *anyone can run the whole project immediately* and
> still get physically sensible data. If you later install ngspice, the same
> script automatically uses the real simulator instead — no code changes.

This station writes 3 CSV spreadsheets (one per cell): `data/raw/inv_sweep.csv`,
`nand2_sweep.csv`, `dff_sweep.csv` — about **4,000 rows each, 12,000 total**.

### Station 2 — Clean and prepare the data
*File: `data/process_data.py`*

Raw numbers aren't ideal for an AI. We do three things:

1. **Feature engineering** — we give the AI not just the raw inputs but some
   helpful "hint" combinations that match the physics, e.g. `vdd²`,
   `1/drive_strength`, and `vdd − threshold_voltage`. (Like giving a student
   the right formula sheet.)
2. **Scaling** — voltage is ~1.8, temperature is ~27, capacitance is tiny. These
   wildly different sizes confuse a neural net, so we **standardize** them all to
   a similar range (this is the `StandardScaler`, saved as `scalers.pkl`).
3. **Log transform** — delays span a 10× range, so we predict `log(delay)`
   instead of raw delay. This makes the math smoother and more accurate.

We also split the data: **70% to train on, 15% to tune with, 15% kept hidden for
the final honest test** (so the AI can't cheat by memorizing).

Output: `data/processed/timing_dataset.csv` and `scalers.pkl`.

### Station 3 — Train the AI
*Files: `ml/model.py` (the brain's shape) + `ml/train.py` (the teaching)*

The model is a small **neural network** — imagine a series of layers of
"neurons" where each connection has a tunable knob ("weight"). Training means:

1. Show it an example's inputs, let it guess the delay.
2. Measure how wrong it was (the "loss").
3. Nudge all the knobs slightly to be less wrong.
4. Repeat for thousands of examples, many times ("epochs").

We use sensible modern settings (AdamW optimizer, a learning-rate schedule, a
robust loss that ignores weird outliers, and "early stopping" so it quits when
it stops improving). It saves the best version to `ml/checkpoints/best_model.pt`
and a graph of its progress to `loss_curve.png`.

> **Analogy:** It's like a student doing thousands of practice problems with an
> answer key, gradually getting better, and we keep the version that scored
> best on the practice exam.

### Station 4 — Test the AI honestly
*File: `ml/evaluate.py`*

Now we use the **hidden 15%** the model never saw and check how close its
predictions are. We report three scores:

- **MAE** (Mean Absolute Error) — on average, how many picoseconds off? → **0.94 ps**
- **MAPE** (error as a percentage) → **1.92%**
- **R²** (how well it captures the pattern; 1.0 = perfect) → **0.9989**

We also benchmark speed: SPICE ≈ a few seconds per point; our AI ≈ 7.6
microseconds per point → about **520,000× faster**. Plus it draws charts
(scatter of predicted-vs-true, error histogram).

### Station 5 — Use the AI to make predictions
*File: `ml/inference.py`*

This is the reusable "prediction engine" (`TimingInference`). You hand it
conditions (cell, voltage, temperature, load, drive, corner) and it returns the
predicted delay in microseconds. It makes sure to apply the *exact same*
feature-prep and scaling as during training — otherwise predictions would be
garbage. It can predict one point, a whole batch, or a full 60-row corner table.

### Station 6 — The website
*Files: `web/app.py` (backend) + `web/templates/index.html`, `web/static/...`*

A web dashboard that loads the trained AI once and lets anyone interact with it
through sliders and buttons. (Full tour in Part 4.)

### Bonus station — The Liberty file
*File: `data/gen_liberty.py`*

This takes the AI's predictions and writes them into a real **`.lib` file** —
the actual industry format foundries ship to chip designers. This shows the
project produces *real, usable engineering output*, not just a demo.

---

## PART 4 — THE WEBSITE, SECTION BY SECTION

Open it at **http://127.0.0.1:5000**. It's a single dark-themed page with six
sections. Here's what each does and what to say about it.

**1. Hero header (top banner)**
Three headline cards: "< 2% error", "~500,000× faster", "3 cell types." This is
your elevator pitch in three numbers.

**2. Interactive Timing Predictor** ⭐ *the star of the demo*
- Left panel: choose the cell (INV/NAND2/DFF), drag sliders for **voltage**,
  **temperature**, **load capacitance**, pick **drive strength** and **process
  corner**.
- Right panel: the predicted delay appears instantly with an animated counter,
  plus tpHL/tpLH, the prediction speed, what SPICE would have taken, and the
  speedup.
- **Demo line:** "Watch — as I raise the voltage, the delay drops, exactly as
  real physics says. And it's predicting in microseconds."

**3. Parameter Sweep Visualizer**
Pick one variable to sweep (e.g. temperature) and it plots a curve of delay vs.
that variable. You can overlay multiple process corners as separate lines.
- **Demo line:** "Here's delay vs. temperature across all process corners — you
  can see SS (slow) is always on top, FF (fast) at the bottom."

**4. PVT Corner Table**
Generates the full 60-combination table (3 voltages × 4 temps × 5 process
corners), **color-coded green (fast) → red (slow)**, with a CSV export button.
- **Demo line:** "This is essentially what a characterization tool outputs —
  the worst-case (slowest, red) corner is what engineers sign off against."

**5. Model Accuracy Dashboard**
Shows the honest test scores (MAE/MAPE/R²), a scatter plot (perfect predictions
land on the diagonal line), an error histogram (tightly centered on 0%), and a
per-corner accuracy breakdown.
- **Demo line:** "Every dot sits on the diagonal — the AI agrees with the
  physics across the board."

**6. Project Explainer + Liberty download**
Three cards summarizing the VLSI, SPICE, and ML pieces, plus a button to
download the generated `.lib` file.

### How the website works under the hood (simple)
Your browser (the **frontend**: HTML/CSS/JavaScript) sends a small request to
the Python **backend** (`app.py`, built with Flask). The backend asks the
**AI engine** (`inference.py`) for a prediction and sends the number back, which
JavaScript displays. The model is loaded **once** when the server starts, so
every request after that is fast.

The backend offers these "endpoints" (URLs the frontend calls):

| URL | What it returns |
|---|---|
| `/api/predict` | one delay prediction |
| `/api/sweep` | a curve of delay vs. one swept variable |
| `/api/corner_table` | the 60-row PVT table |
| `/api/model_metrics` | the accuracy scores |
| `/api/dataset_stats` | dataset size summary |
| `/api/liberty` | downloads the generated .lib file |
