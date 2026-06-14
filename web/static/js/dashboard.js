/* ML-Assisted VLSI Timing dashboard front-end */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

Chart.defaults.color = "#94a3b8";
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
Chart.defaults.font.family = "JetBrains Mono, monospace";

const AMBER = "#f59e0b";
const CELL_COLORS = { inv: "#f59e0b", nand2: "#10b981", dff: "#60a5fa" };
const CORNER_COLORS = { tt: "#f59e0b", ff: "#10b981", ss: "#ef4444", fs: "#60a5fa", sf: "#a78bfa" };

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = msg;
  $("#toast-container").appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* animated number counter */
function animateNumber(el, to, suffix = "", dur = 400) {
  const from = parseFloat(el.dataset.val || "0");
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const v = from + (to - from) * eased;
    el.textContent = v.toFixed(1) + suffix;
    if (p < 1) requestAnimationFrame(step);
    else el.dataset.val = to;
  }
  requestAnimationFrame(step);
}

class PredictorUI {
  constructor() {
    this.state = { cell: "inv", vdd: 1.8, temp: 27, cload_ff: 10, drive: 2, corner: "tt" };
    this.sweepCorners = new Set(["tt"]);
    this.lastCornerRows = [];
    this.bindInputs();
    this.initCharts();
    this.predict();
    this.loadMetrics();
    this.loadStats();
  }

  bindInputs() {
    const dbPredict = debounce(() => this.predict(), 50);

    $$("#cell-toggle .toggle").forEach((b) =>
      b.addEventListener("click", () => {
        $$("#cell-toggle .toggle").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        this.state.cell = b.dataset.cell;
        this.predict();
      }));

    $$("#drive-toggle .toggle").forEach((b) =>
      b.addEventListener("click", () => {
        $$("#drive-toggle .toggle").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        this.state.drive = parseInt(b.dataset.drive, 10);
        this.predict();
      }));

    const vdd = $("#vdd"), temp = $("#temp"), cload = $("#cload"), corner = $("#corner");
    vdd.addEventListener("input", () => { this.state.vdd = +vdd.value; $("#vdd-val").textContent = (+vdd.value).toFixed(2) + " V"; dbPredict(); });
    temp.addEventListener("input", () => { this.state.temp = +temp.value; $("#temp-val").textContent = temp.value + " °C"; dbPredict(); });
    cload.addEventListener("input", () => { this.state.cload_ff = +cload.value; $("#cload-val").textContent = (+cload.value).toFixed(1) + " fF"; dbPredict(); });
    corner.addEventListener("change", () => { this.state.corner = corner.value; this.predict(); });

    $("#predict-btn").addEventListener("click", () => this.predict());
    document.addEventListener("keydown", (e) => { if (e.key === "Enter") this.predict(); });

    // sweep
    $$("#sweep-corners .chip").forEach((c) =>
      c.addEventListener("click", () => {
        c.classList.toggle("active");
        const k = c.dataset.c;
        if (c.classList.contains("active")) this.sweepCorners.add(k);
        else this.sweepCorners.delete(k);
      }));
    $("#sweep-btn").addEventListener("click", () => this.runSweep());

    // corner table
    $("#ct-btn").addEventListener("click", () => this.cornerTable());
    $("#ct-csv").addEventListener("click", () => this.exportCSV());
  }

  async predict() {
    try {
      const res = await apiPost("/api/predict", this.state);
      animateNumber($("#tpd-result"), res.tpd_ps, "");
      $("#tphl-result").textContent = res.tpHL_ps.toFixed(1) + " ps";
      $("#tplh-result").textContent = res.tpLH_ps.toFixed(1) + " ps";
      $("#latency-result").textContent = res.latency_us.toFixed(2) + " µs";
      $("#spice-result").textContent = "~" + res.spice_equiv_s.toFixed(1) + " s";
      $("#speedup-result").textContent = res.speedup.toLocaleString() + "×";
    } catch (e) {
      toast("Predict failed: " + e.message, true);
    }
  }

  initCharts() {
    this.sweepChart = new Chart($("#sweep-chart"), {
      type: "line",
      data: { datasets: [] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
        interaction: { mode: "nearest", intersect: false },
        scales: { x: { type: "linear", title: { display: true, text: "swept value" } },
          y: { title: { display: true, text: "tpd (ps)" } } },
        plugins: { legend: { position: "top" } },
      },
    });

    this.scatterChart = new Chart($("#scatter-chart"), {
      type: "scatter",
      data: { datasets: [] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: "SPICE delay (ps)" } },
          y: { title: { display: true, text: "ML predicted (ps)" } } },
      },
    });

    this.errorChart = new Chart($("#error-chart"), {
      type: "bar",
      data: { labels: [], datasets: [] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { title: { display: true, text: "error (%)" } },
          y: { title: { display: true, text: "count" } } },
        plugins: { legend: { display: false } },
      },
    });
  }

  async runSweep() {
    const param = $("#sweep-param").value;
    const cell = $("#sweep-cell").value;
    const drive = parseInt($("#sweep-drive").value, 10);
    const corners = Array.from(this.sweepCorners);
    if (corners.length === 0) { toast("Select at least one corner", true); return; }
    try {
      const datasets = [];
      for (const c of corners) {
        const res = await apiPost("/api/sweep", {
          cell, drive, cload_ff: this.state.cload_ff, corner: c, sweep_param: param });
        datasets.push({
          label: c.toUpperCase(),
          data: res.data.map((p) => ({ x: p.param_value, y: p.tpd_ps })),
          borderColor: CORNER_COLORS[c], backgroundColor: CORNER_COLORS[c],
          pointRadius: 0, borderWidth: 2, tension: 0.25,
        });
      }
      this.sweepChart.data.datasets = datasets;
      this.sweepChart.options.scales.x.title.text = param;
      this.sweepChart.update();
    } catch (e) { toast("Sweep failed: " + e.message, true); }
  }

  async cornerTable() {
    const cell = $("#ct-cell").value;
    const drive = $("#ct-drive").value;
    const cload = $("#ct-cload").value;
    try {
      const res = await apiGet(`/api/corner_table?cell=${cell}&drive=${drive}&cload_ff=${cload}`);
      this.lastCornerRows = res.rows;
      this.renderCornerTable(res);
    } catch (e) { toast("Corner table failed: " + e.message, true); }
  }

  renderCornerTable(res) {
    const rows = res.rows;
    const corners = res.corners;
    const vdds = [...new Set(rows.map((r) => r.vdd))].sort((a, b) => a - b);
    const temps = [...new Set(rows.map((r) => r.temp))].sort((a, b) => a - b);
    const vals = rows.map((r) => r.tpd_ps);
    const min = Math.min(...vals), max = Math.max(...vals);
    const color = (v) => {
      const t = (v - min) / (max - min + 1e-9);
      const r = Math.round(16 + t * (239 - 16));
      const g = Math.round(185 - t * (185 - 68));
      const b = Math.round(129 - t * (129 - 68));
      return `rgba(${r},${g},${b},0.35)`;
    };
    const lookup = {};
    rows.forEach((r) => { lookup[`${r.vdd}|${r.temp}|${r.corner}`] = r.tpd_ps; });

    let html = "<thead><tr><th data-k='vdd'>VDD (V)</th><th data-k='temp'>Temp (°C)</th>";
    corners.forEach((c) => html += `<th>${c.toUpperCase()}</th>`);
    html += "</tr></thead><tbody>";
    vdds.forEach((v) => temps.forEach((t) => {
      html += `<tr><td>${v.toFixed(2)}</td><td>${t}</td>`;
      corners.forEach((c) => {
        const val = lookup[`${v}|${t}|${c}`];
        html += `<td class="cell-val" style="background:${color(val)}">${val.toFixed(1)}</td>`;
      });
      html += "</tr>";
    }));
    html += "</tbody>";
    $("#corner-table").innerHTML = html;
  }

  exportCSV() {
    if (!this.lastCornerRows.length) { toast("Generate a table first", true); return; }
    let csv = "vdd,temp,corner,tpd_ps\n";
    this.lastCornerRows.forEach((r) => csv += `${r.vdd},${r.temp},${r.corner},${r.tpd_ps}\n`);
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "corner_table.csv";
    a.click();
    toast("CSV exported");
  }

  async loadMetrics() {
    try {
      const m = await apiGet("/api/model_metrics");
      $("#hero-mape").textContent = "< " + Math.max(2, Math.ceil(m.mape_pct)) + "%";
      $("#hero-speed").textContent = Math.round(m.speedup).toLocaleString() + "×";

      $("#stats-row").innerHTML = [
        ["MAE", m.mae_ps.toFixed(2) + " ps"],
        ["MAPE", m.mape_pct.toFixed(2) + " %"],
        ["R²", m.r2.toFixed(4)],
        ["speedup", Math.round(m.speedup).toLocaleString() + "×"],
        ["train", m.n_train.toLocaleString()],
        ["test", m.n_test.toLocaleString()],
      ].map(([l, v]) => `<div class="stat-box"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

      // scatter
      const byCell = {};
      m.scatter.forEach((p) => { (byCell[p.cell] = byCell[p.cell] || []).push({ x: p.spice, y: p.ml }); });
      this.scatterChart.data.datasets = Object.entries(byCell).map(([cell, pts]) => ({
        label: cell.toUpperCase(), data: pts,
        backgroundColor: CELL_COLORS[cell] || "#999", pointRadius: 2.5,
      }));
      this.scatterChart.update();

      // error histogram
      const errs = m.error_pct;
      const nb = 40;
      const lo = Math.min(...errs), hi = Math.max(...errs);
      const w = (hi - lo) / nb || 1;
      const counts = new Array(nb).fill(0);
      errs.forEach((e) => { let i = Math.floor((e - lo) / w); if (i >= nb) i = nb - 1; if (i < 0) i = 0; counts[i]++; });
      this.errorChart.data.labels = counts.map((_, i) => (lo + (i + 0.5) * w).toFixed(1));
      this.errorChart.data.datasets = [{ data: counts, backgroundColor: AMBER }];
      this.errorChart.update();

      // breakdown
      const b = m.corner_breakdown;
      let html = "<thead><tr><th>cell</th><th>corner</th><th>n</th><th>MAE (ps)</th><th>MAPE (%)</th><th>R²</th></tr></thead><tbody>";
      b.forEach((r) => html += `<tr><td>${r.cell}</td><td>${r.proc.toUpperCase()}</td><td>${r.n}</td><td>${r.mae_ps}</td><td>${r.mape_pct}</td><td>${r.r2}</td></tr>`);
      $("#breakdown-table").innerHTML = html + "</tbody>";
    } catch (e) {
      toast("Metrics unavailable — run `make evaluate`", true);
    }
  }

  async loadStats() {
    try { await apiGet("/api/dataset_stats"); } catch (e) { /* optional */ }
  }
}

window.addEventListener("DOMContentLoaded", () => { window.ui = new PredictorUI(); });
