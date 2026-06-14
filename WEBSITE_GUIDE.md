# WEBSITE GUIDE — How To Use The Dashboard (Complete Walkthrough)

> This explains **every button, slider, number, chart, and color** on the
> dashboard: what it does, what happens when you change it, what the numbers
> mean, what you can conclude from them, and how it maps to the real world.
> Open the site at **http://127.0.0.1:5000** and follow along.

---

**The page has 6 sections, top to bottom:**
1. Hero header (the headline numbers)
2. Interactive Timing Predictor ⭐ (the main tool)
3. Parameter Sweep Visualizer (graphs)
4. PVT Corner Table (the color grid)
5. Model Accuracy Dashboard (proof it works)
6. Project Explainer (+ Liberty download)

---

## QUICK REFRESHER: WHAT IS "DELAY"?

Everything on this site revolves around one number: **propagation delay (tpd)**,
measured in **picoseconds (ps)** = trillionths of a second.

> **Delay = how long a logic gate takes to react after its input changes.**

Smaller delay = faster gate = the chip can run at a higher clock speed.
A real-world hook: if every gate in a chip's critical path is 10% slower, the
whole processor must run ~10% slower (lower GHz). So these picoseconds directly
decide how many gigahertz a chip can hit.

---

## SECTION 1 — HERO HEADER

At the very top you see three big cards:

| Card | Meaning |
|---|---|
| **< 2% error** | The AI's predictions are within ~2% of the "true" physics answer. |
| **~500,000× faster** | The AI predicts a delay ~half a million times faster than SPICE simulation. |
| **3 cell types** | It handles three building blocks: INV, NAND2, DFF. |

The first two numbers are pulled **live** from the trained model's real test
results — they're not hard-coded. This section is your **30-second pitch**.

---

## SECTION 2 — INTERACTIVE TIMING PREDICTOR

This is the heart of the demo. You set conditions on the **left**, and the AI's
predicted delay appears on the **right** — instantly, as you move things.

### The controls (left panel) and what each one does

#### ▸ Cell type: INV / NAND2 / DFF
Picks which building block you're testing.
- **INV (inverter)** — simplest, fastest. Baseline.
- **NAND2** — a 2-input gate, ~**1.35× slower** than INV (it has two transistors
  stacked in series, which slows it down).
- **DFF (flip-flop)** — a memory element, ~**2.8× slower** than INV (it's built
  from ~12 transistors).

**What happens when you change it:** the predicted delay jumps. Switch INV →
NAND2 → DFF at the same settings and watch the number grow each time.
**What you infer:** more complex gates are slower. This is why chip designers
prefer simple gates on speed-critical paths.

#### ▸ VDD slider (1.62 V → 1.98 V)
This is the **power supply voltage**. Nominal is 1.80 V; the range is ±10%
(what a real power rail might droop or spike to).

**What happens when you slide it UP:** the delay goes **DOWN** (gate gets
faster). Slide it down → delay goes up.
**Why:** higher voltage pushes current through the transistors harder, charging
the output faster.
**Real-life link:** this is exactly why "overclockers" *raise* CPU voltage to
hit higher speeds — and why laptops *lower* voltage to save battery (accepting
slower, lower-clock operation). It's the speed-vs-power trade-off in one slider.

#### ▸ Temperature slider (−40 °C → 125 °C)
The operating temperature of the chip.

**What happens when you slide it UP (hotter):** delay goes **UP** (gate gets
slower).
**Why:** heat makes electrons bump into the silicon lattice more (mobility
drops), so current flows less easily.
**Real-life link:** a phone or GPU that gets hot literally runs slower — this is
**thermal throttling**. The −40 °C end represents extreme cold (automotive,
aerospace); 125 °C represents a hot server or car engine bay.

#### ▸ Load capacitance slider (1 fF → 50 fF)
"Load" = how much stuff this gate has to drive (the wires and gate inputs
connected to its output). Measured in **femtofarads (fF)**.

**What happens when you slide it UP (heavier load):** delay goes **UP**, almost
linearly.
**Why:** a bigger capacitor takes longer to charge/discharge. Driving more
downstream gates = more load = slower.
**Real-life link:** a gate connected to a long wire or many other gates ("high
fanout") is slower. This is why chip layout and buffering matter so much.

#### ▸ Drive strength: 1× / 2× / 4× / 8×
"Drive strength" = how big (wide) the transistors are. A 4× cell has 4× wider
transistors than a 1× cell.

**What happens when you increase it:** delay goes **DOWN** (roughly halves each
time you double it: 1×→2× ≈ half, 2×→4× ≈ half again).
**Why:** wider transistors push more current, charging the load faster.
**Real-life link:** standard-cell libraries literally ship multiple sizes of the
same gate (INV_X1, INV_X2, INV_X4...). Designers pick a bigger one to fix a slow
path — but bigger cells use more power and area. Another classic trade-off.

#### ▸ Process corner: TT / FF / SS / FS / SF
Accounts for **manufacturing variation** — no two chips come out identical.
- **TT** = Typical (the average chip)
- **FF** = Fast-Fast (both transistor types came out fast) → **fastest**
- **SS** = Slow-Slow → **slowest**
- **FS / SF** = mixed (one type fast, one slow)

**What happens when you switch TT → SS:** delay goes up (~20% slower).
Switch TT → FF: delay drops (~15% faster).
**What you infer:** the *same design* performs differently depending on
manufacturing luck. Engineers must guarantee it works even on a "bad" (SS) chip.
**Real-life link:** this is **silicon binning** — chips that happen to come out
"FF" are sold as higher-clock (more expensive) parts; "SS" chips are sold as
lower-clock parts. Same wafer, different price tags.

#### ▸ Predict button (or press Enter)
Forces a prediction. You usually don't need it — the sliders predict **live** as
you move them (there's a tiny 50-millisecond debounce so it feels smooth).

### The results (right panel) and what every number means

| Display | Meaning | What to look at |
|---|---|---|
| **Big number + "ps"** | Predicted **tpd** (average propagation delay). The headline answer. | Counts up with a quick animation each time it changes. |
| **tpHL** | Delay when output goes **High → Low**. | Usually slightly different from tpLH. |
| **tpLH** | Delay when output goes **Low → High**. | tpd is the average of these two. |
| **ML latency** | How long *the AI* took to compute this prediction (microseconds, µs). | This is the "fast" part — the whole point. |
| **SPICE equiv.** | How long the *old simulator* would have taken for the same point (seconds). | The "slow" baseline we're beating. |
| **Speedup** | SPICE time ÷ AI time. | The headline efficiency win. |

> **One honest note about "ML latency":** the *very first* prediction after the
> page loads can show a few milliseconds — that's one-time Python/PyTorch
> warm-up, not the model's real speed. Move a slider once and it drops to the
> microsecond range. The true sustained speed is the **7.6 µs/point** measured
> in the batch benchmark (Section 5). Mention this if asked — it's the honest
> framing.

### Try this mini-experiment (great for a demo)
1. Set INV, 1.80 V, 27 °C, 10 fF, drive 2×, TT. Note the delay (~15 ps).
2. Drag VDD up to 1.98 V → delay drops. Drag down to 1.62 V → delay rises.
3. Switch corner to **SS** → slower. To **FF** → faster.
4. Switch cell to **DFF** → delay roughly triples.
Every change behaves the way real physics says it should — that's the proof the
AI learned the physics, not just memorized numbers.

---

## SECTION 3 — PARAMETER SWEEP VISUALIZER (the graphs)

Instead of one prediction at a time, this **sweeps one variable across its whole
range** and plots a curve. It answers "how does delay *trend* as I change X?"

### The controls
- **Sweep** dropdown — which variable to vary: **VDD**, **Temperature**, or
  **Load capacitance**. (Everything else is held fixed.)
- **Cell** — which gate.
- **Drive** — transistor size.
- **Corners** chips — click to select one or more process corners. Each selected
  corner becomes a **separate line** on the graph.
- **Run sweep** button — generates ~50 predictions across the range and plots
  them.

### Reading the chart
- **X-axis** = the variable you're sweeping (e.g. temperature).
- **Y-axis** = predicted delay (ps).
- **Each line** = one process corner. Hover any point to see exact values.

### What the curves tell you
- **Sweep VDD:** the line **slopes downward** — higher voltage, lower delay.
  The curve is slightly bent (non-linear), reflecting real transistor physics.
- **Sweep Temperature:** the line **slopes upward** — hotter = slower.
- **Sweep Load:** the line is **nearly straight, rising** — delay ∝ load.
- **Multiple corners:** the **SS line sits highest** (slowest), **FF lowest**
  (fastest), with TT in between. The vertical gap between SS and FF shows you
  *how much manufacturing variation costs you* in speed.

### Real-life use
This is exactly the kind of plot a timing engineer makes to decide things like:
"If this chip must work up to 125 °C on a slow (SS) part, what's my worst-case
delay, and will it still meet my clock target?" The gap between the FF and SS
curves is the **margin** the whole design must tolerate.

---

## SECTION 4 — PVT CORNER TABLE (the color grid)

Click **Generate corner table** and you get the full grid: **60 combinations** =
3 voltages × 4 temperatures × 5 process corners, for the chosen cell/drive/load.

### How to read it
- **Rows** = a (VDD, Temperature) pairing.
- **Columns** = the 5 process corners (TT/FF/SS/FS/SF).
- **Each cell** = predicted delay in ps for that exact combination.

### The colors (this is the key feature)
Cells are color-coded from **green = fastest** to **red = slowest**, scaled
across this table.
- **Green cluster** → high voltage, cold, FF corner (best case).
- **Red cluster** → low voltage, hot, SS corner (worst case).

**What you infer at a glance:** the single **reddest cell is your worst-case
corner** — the slowest the gate will *ever* be. That's the number engineers must
design around (**sign-off**). The greenest is the best case.

### Export CSV button
Downloads the table as a spreadsheet you can open in Excel — handy for reports
or feeding into other tools.

### Real-life link
This grid *is* a miniature version of what professional characterization tools
produce. For a real chip, every one of hundreds of cells gets a table like this
(actually bigger — also varying input speed and load). Multiply that out and you
see why it's millions of simulations and why a fast AI shortcut is valuable.

---

## SECTION 5 — MODEL ACCURACY DASHBOARD (proof it works)

This section proves the AI is trustworthy by testing it on data it **never saw
during training** (the hidden 15%).

### The stats row
| Stat | Meaning | Good value | What it tells you |
|---|---|---|---|
| **MAE** | Mean Absolute Error — average miss, in ps | **0.94 ps** | On average the AI is less than 1 ps off. |
| **MAPE** | Mean Absolute % Error | **1.92%** | Typical prediction is within ~2% of truth. |
| **R²** | "R-squared," how well it captures the pattern (1.0 = perfect) | **0.9989** | It explains 99.89% of the variation — essentially flawless. |
| **speedup** | AI speed advantage over SPICE | **~520,000×** | The efficiency headline. |
| **train** | # examples it learned from | 8,400 | — |
| **test** | # hidden examples it was graded on | 1,800 | Honest, unseen evaluation. |

### The scatter chart (left): "SPICE vs ML predicted delay"
- **X-axis** = the true delay; **Y-axis** = the AI's predicted delay.
- Each dot is one test example, colored by cell type.
- **The diagonal line = perfect prediction.**
**What to look for:** all dots hug the diagonal tightly → the AI agrees with the
truth everywhere, for all three cells and across the full delay range.

### The error histogram (right): "Prediction-error distribution"
- **X-axis** = prediction error in %; **Y-axis** = how many predictions had that
  error.
- **What to look for:** a tall, narrow bell centered on **0%**. That means most
  predictions are nearly spot-on and errors are small and unbiased (not
  systematically too high or too low).

### Per-corner accuracy table
Breaks the error down by cell and process corner, so you can confirm the AI is
accurate **everywhere**, not just on average. (You'll see DFF is slightly harder
than INV — it's the most complex cell — but still well within target.)

### Real-life use
This is the section that convinces a skeptic. In industry, before anyone trusts
an ML surrogate to replace SPICE, they demand exactly this: error metrics on
held-out data, a correlation plot, and a per-corner breakdown to make sure it
doesn't fail at a specific worst-case condition.

---

## SECTION 6 — PROJECT EXPLAINER + LIBERTY DOWNLOAD

Three summary cards (VLSI / SPICE / ML) for visitors, plus two buttons:
- **View GitHub** — placeholder link for your repo.
- **Download .lib** — downloads `sky130_ml_char.lib`, the **real industry-format
  timing file** the AI generated.

### What's a `.lib` file and why it matters
A **Liberty (`.lib`) file** is the actual text format chip foundries (TSMC,
Samsung, Intel...) ship to designers. It contains lookup tables of cell delay
vs. input speed and output load. Every digital-design tool on Earth reads these
files. By generating one, this project shows it produces **real, usable
engineering output** — not just a pretty demo. Open it in Notepad: you'll see
the delay tables the AI predicted, in professional syntax.

---

## PUTTING IT ALL TOGETHER: A 2-MINUTE LIVE DEMO SCRIPT

1. **(Hero)** "This is an AI that predicts chip-gate timing — under 2% error,
   ~500,000× faster than the physics simulator it replaces."
2. **(Predictor)** "Watch the delay update live as I move the voltage slider —
   higher voltage, lower delay, exactly like real silicon. Now I'll switch to a
   slow manufacturing corner — slower. And to a flip-flop — much slower."
3. **(Sweep)** "Here's delay vs. temperature across all corners. The slow corner
   is always on top — that gap is the margin a real design must tolerate."
4. **(Corner table)** "This color grid is the worst-case map — the reddest cell
   is what engineers sign off against. I can export it to CSV."
5. **(Accuracy)** "And here's the proof: on data it never saw, every dot sits on
   the perfect-prediction line, errors center on zero."
6. **(Liberty)** "Finally, it exports a real Liberty file — the exact format
   foundries ship to chip designers."

---

## TROUBLESHOOTING

- **Numbers show "--" or "Metrics unavailable":** the model/results files are
  missing. Run `python run_all.py` first, then restart `python web/app.py`.
- **Page won't load:** make sure the server terminal is still running and you're
  on `http://127.0.0.1:5000` (not `https`).
- **Charts are empty:** click the **Run sweep** / **Generate corner table**
  buttons — those sections load on demand.
- **First prediction seems slow:** that's one-time warm-up; move a slider and
  it's instant after that.
```
