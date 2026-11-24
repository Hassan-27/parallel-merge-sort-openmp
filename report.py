"""
report.py - builds PDC_Project_Report.pdf from the benchmark results.

    python report.py

Reads results/bench.csv, results/cutoff.csv and results/test.txt (written by
`mergesort bench` and `mergesort test`), computes speedup, efficiency and the
performance model, draws the graphs and puts everything into one PDF
(HTML printed to PDF with headless Edge/Chrome). Graphs are embedded in the
report directly; a summary table is also written to results/summary.csv.
"""
import base64
import csv
import io
import math
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
PDF = ROOT / "PDC_Project_Report.pdf"
GITHUB_URL = "https://github.com/Hassan-27/parallel-merge-sort-openmp"
CORES = 4  # physical cores of the test machine (threads 5-8 are Hyper-Threads)

# ============================================================================
# 1. LOAD RESULTS
# ============================================================================

def read_csv(name):
    with open(RES / name, newline="") as f:
        return [dict(r, n=int(r["n"]), threads=int(r["threads"]), cutoff=int(r["cutoff"]),
                     time_s=float(r["time_s"])) for r in csv.DictReader(f)]

bench = read_csv("bench.csv")
cut = read_csv("cutoff.csv") if (RES / "cutoff.csv").exists() else []
T = {(r["algo"], r["n"], r["threads"]): r["time_s"] for r in bench}
sizes = sorted({r["n"] for r in bench})
threads = sorted({r["threads"] for r in bench if r["algo"].startswith("par")})
cutoff = bench[0]["cutoff"]
nmax, pmax = sizes[-1], max(threads)


def fmt_n(n):
    for div, suf in ((10**9, "G"), (10**6, "M"), (10**3, "K")):
        if n >= div and n % div == 0:
            return f"{n // div}{suf}"
    return f"{n:,}"

# ============================================================================
# 2. PERFORMANCE MODEL (work/span + Amdahl's law)
#
#   D      = log_k(n)                        levels of the recursion tree
#   L      = T_seq / D                       cost of one level (each level merges all n elements)
#   T_span = L * (1 + 1/k + 1/k^2 + ...)     = L * k / (k - 1)   (merges on the path leaf -> root)
#   T_p    = T_span + (T_seq - T_span) / min(p, CORES)
# ============================================================================

def span(way, n):
    t_seq = T[(f"seq{way}", n, 1)]
    return t_seq / math.log(n, way) * way / (way - 1)


def predict(way, n, p):
    t_seq = T[(f"seq{way}", n, 1)]
    s = span(way, n)
    return s + (t_seq - s) / min(p, CORES)


rows = []  # one row per parallel/sequential measurement
for n in sizes:
    for way in (2, 4):
        seq = T[(f"seq{way}", n, 1)]
        rows.append(dict(algo=f"seq{way}", n=n, threads=1, time=seq, speedup=1.0, eff=1.0, model=seq, err=0.0))
        for p in threads:
            t = T[(f"par{way}", n, p)]
            m = predict(way, n, p)
            rows.append(dict(algo=f"par{way}", n=n, threads=p, time=t, speedup=seq / t, eff=seq / t / p,
                             model=m, err=100 * (m - t) / t))


def get(algo, n, p=1):
    return next(r for r in rows if r["algo"] == algo and r["n"] == n and r["threads"] == p)


with open(RES / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algo", "n", "threads", "time_s", "speedup", "efficiency", "model_s", "model_err_pct"])
    for r in rows:
        w.writerow([r["algo"], r["n"], r["threads"], f"{r['time']:.4f}", f"{r['speedup']:.3f}",
                    f"{r['eff']:.3f}", f"{r['model']:.4f}", f"{r['err']:.1f}"])

# model accuracy over runs with 2+ threads and >= 0.1 s (at 1 thread the model is just T_seq)
checked = [r for r in rows if r["algo"].startswith("par") and r["threads"] > 1 and r["time"] >= 0.1]
mape = sum(abs(r["err"]) for r in checked) / len(checked)
within15 = sum(abs(r["err"]) <= 15 for r in checked) / len(checked)


def mape_of(algo, many_threads):
    rr = [abs(r["err"]) for r in checked if r["algo"] == algo and (r["threads"] >= CORES) == many_threads]
    return sum(rr) / len(rr) if rr else 0.0

# ============================================================================
# 3. HEADLINE NUMBERS
# ============================================================================

s2, s4 = get("seq2", nmax), get("seq4", nmax)
p2, p4 = get("par2", nmax, pmax), get("par4", nmax, pmax)
best2 = max((get("par2", nmax, p) for p in threads), key=lambda r: r["speedup"])
best4 = max((get("par4", nmax, p) for p in threads), key=lambda r: r["speedup"])
best_par = min((get(a, nmax, p) for a in ("par2", "par4") for p in threads), key=lambda r: r["time"])
best_seq = min(s2["time"], s4["time"])
seq_faster = "4-way" if s4["time"] < s2["time"] else "2-way"
seq_gap = 100 * abs(s2["time"] - s4["time"]) / max(s2["time"], s4["time"])
seq_wins4 = sum(get("seq4", n)["time"] < get("seq2", n)["time"] for n in sizes)
par_cfgs = [(n, p) for n in sizes for p in threads if p > 1]
wins4 = sum(get("par4", n, p)["time"] < get("par2", n, p)["time"] for n, p in par_cfgs)
f2 = span(2, nmax) / s2["time"]
f4 = span(4, nmax) / s4["time"]

cut_best, cut_small, near_opt = {}, {}, {}
for way in (2, 4):
    rr = [r for r in cut if r["algo"] == f"par{way}"]
    if rr:
        cut_best[way] = min(rr, key=lambda r: r["time_s"])
        cut_small[way] = min(rr, key=lambda r: r["cutoff"])
        ok = [r["cutoff"] for r in rr if r["time_s"] <= 1.10 * cut_best[way]["time_s"]]
        near_opt[way] = (min(ok), max(ok))

# ============================================================================
# 4. GRAPHS (embedded in the report as base64 PNG, no files written)
# ============================================================================

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK2, GRID = "#52514e", "#e4e3df"
WAY = {2: BLUE, 4: ORANGE}
SIZE_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 2, "lines.markersize": 5.5, "legend.frameon": False, "legend.fontsize": 8.5,
})
plot_sizes = [n for n in sizes if n in (1_000_000, 10_000_000, 50_000_000, 100_000_000)] + [nmax]
plot_sizes = sorted(set(plot_sizes))[:6]


def png(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_time_vs_threads():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for way in (2, 4):
        ax.plot(threads, [get(f"par{way}", nmax, p)["time"] for p in threads], marker="o", color=WAY[way],
                label=f"{way}-way measured")
        ax.plot(threads, [predict(way, nmax, p) for p in threads], ls=":", marker="s", mfc="white",
                color=WAY[way], label=f"{way}-way model")
        ax.axhline(get(f"seq{way}", nmax)["time"], color=WAY[way], lw=1, ls="--", alpha=0.6)
    ax.set(title=f"Execution time, n = {fmt_n(nmax)} integers", xlabel="threads", ylabel="time (s)", xticks=threads)
    ax.set_ylim(0, None)
    ax.legend()
    return png(fig)


def fig_per_size(metric):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, way in zip(axes, (2, 4)):
        if metric == "speedup":
            ax.plot([1, pmax], [1, pmax], color=INK2, lw=1, ls="--", label="ideal (linear)")
        for i, n in enumerate(plot_sizes):
            vals = [get(f"par{way}", n, p)[metric] * (100 if metric == "eff" else 1) for p in threads]
            ax.plot(threads, vals, marker="o", color=SIZE_COLORS[i], label=f"n = {fmt_n(n)}")
        ax.axvline(CORES, color=GRID, lw=6, zorder=0)
        name = "speedup" if metric == "speedup" else "parallel efficiency"
        ax.set(title=f"{way}-way merge sort: {name}", xlabel="threads", xticks=threads)
        ax.set_ylim(0, pmax + 0.5 if metric == "speedup" else 120)
    axes[0].set_ylabel("speedup (T_seq / T_par)" if metric == "speedup" else "efficiency (%)")
    axes[1].legend(loc="upper left" if metric == "speedup" else "upper right")
    return png(fig)


def fig_time_vs_n():
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for way in (2, 4):
        ax.plot(sizes, [get(f"seq{way}", n)["time"] for n in sizes], marker="o", ls="--", color=WAY[way],
                label=f"{way}-way sequential")
        ax.plot(sizes, [get(f"par{way}", n, pmax)["time"] for n in sizes], marker="o", color=WAY[way],
                label=f"{way}-way, {pmax} threads")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([fmt_n(n) for n in sizes])
    ax.minorticks_off()
    ax.set(title="Execution time vs array size", xlabel="array size n", ylabel="time (s, log scale)")
    ax.legend()
    return png(fig)


def fig_model_accuracy():
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    lo, hi = 0.08, max(r["time"] for r in checked) * 1.4
    ax.fill_between([lo, hi], [lo * 0.85, hi * 0.85], [lo * 1.15, hi * 1.15], color=GRID, lw=0, label="±15% band")
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", label="perfect prediction")
    for way in (2, 4):
        pts = [r for r in checked if r["algo"] == f"par{way}"]
        ax.scatter([r["time"] for r in pts], [r["model"] for r in pts], s=26, color=WAY[way],
                   edgecolor="white", linewidth=0.8, label=f"{way}-way", zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="measured time (s)",
           ylabel="predicted time (s)", title=f"Model accuracy (mean error {mape:.1f}%)")
    ax.legend(loc="upper left")
    return png(fig)


def fig_cutoff():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for way in (2, 4):
        rr = sorted((r for r in cut if r["algo"] == f"par{way}"), key=lambda r: r["cutoff"])
        ax.plot([r["cutoff"] for r in rr], [r["time_s"] for r in rr], marker="o", color=WAY[way], label=f"{way}-way")
    ax.set_xscale("log", base=2)
    ax.set(title=f"Effect of task cutoff (n = {fmt_n(cut[0]['n'])}, {cut[0]['threads']} threads)",
           xlabel="task cutoff (elements, log scale)", ylabel="time (s)")
    ax.set_ylim(0, None)
    ax.legend()
    return png(fig)


def img(b64, width, caption):
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption>{caption}</figcaption></figure>')

# ============================================================================
# 5. TABLES AND CODE SNIPPETS
# ============================================================================

def results_table(n):
    head = ("<tr><th rowspan=2>Threads</th><th colspan=3>2-way merge sort</th><th colspan=3>4-way merge sort</th>"
            "<th rowspan=2>Faster</th></tr><tr><th>Time (s)</th><th>Speedup</th><th>Efficiency</th>"
            "<th>Time (s)</th><th>Speedup</th><th>Efficiency</th></tr>")
    a, b = get("seq2", n), get("seq4", n)
    body = [f"<tr class=seq><td>Sequential</td><td>{a['time']:.3f}</td><td>1.00</td><td>-</td>"
            f"<td>{b['time']:.3f}</td><td>1.00</td><td>-</td><td>{'4-way' if b['time'] < a['time'] else '2-way'}</td></tr>"]
    for p in threads:
        a, b = get("par2", n, p), get("par4", n, p)
        body.append(f"<tr><td>{p}</td><td>{a['time']:.3f}</td><td>{a['speedup']:.2f}&times;</td>"
                    f"<td>{100 * a['eff']:.0f}%</td><td>{b['time']:.3f}</td><td>{b['speedup']:.2f}&times;</td>"
                    f"<td>{100 * b['eff']:.0f}%</td><td>{'4-way' if b['time'] < a['time'] else '2-way'}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def all_sizes_table():
    head = ("<tr><th>n</th><th>seq 2-way</th><th>seq 4-way</th>"
            + "".join(f"<th>par2 p={p}</th>" for p in (CORES, pmax))
            + "".join(f"<th>par4 p={p}</th>" for p in (CORES, pmax))
            + "<th>best speedup<br>(vs best seq)</th></tr>")
    body = []
    for n in sizes:
        bs = min(get("seq2", n)["time"], get("seq4", n)["time"])
        best = min((get(a, n, p) for a in ("par2", "par4") for p in threads), key=lambda r: r["time"])
        cells = [fmt_n(n), f"{get('seq2', n)['time']:.3f}", f"{get('seq4', n)['time']:.3f}"]
        cells += [f"{get('par2', n, p)['time']:.3f}" for p in (CORES, pmax)]
        cells += [f"{get('par4', n, p)['time']:.3f}" for p in (CORES, pmax)]
        cells.append(f"{bs / best['time']:.2f}&times; ({best['algo']}, p={best['threads']})")
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table class=small>{head}{''.join(body)}</table>"


def model_table():
    head = "<tr><th>Version</th><th>Threads</th><th>Measured (s)</th><th>Model (s)</th><th>Error</th></tr>"
    body = [f"<tr><td>{r['algo']}</td><td>{r['threads']}</td><td>{r['time']:.3f}</td><td>{r['model']:.3f}</td>"
            f"<td>{r['err']:+.1f}%</td></tr>"
            for algo in ("par2", "par4") for r in rows if r["algo"] == algo and r["n"] == nmax and r["threads"] > 1]
    return f"<table class=small>{head}{''.join(body)}</table>"


def cutoff_table():
    cs = sorted({r["cutoff"] for r in cut})
    t = {(r["algo"], r["cutoff"]): r["time_s"] for r in cut}
    body = "".join(f"<tr><td>{c:,}</td><td>{t[('par2', c)]:.3f}</td><td>{t[('par4', c)]:.3f}</td></tr>" for c in cs)
    return f"<table class=small><tr><th>Cutoff (elements)</th><th>2-way (s)</th><th>4-way (s)</th></tr>{body}</table>"


def tests_table():
    p = RES / "test.txt"
    if not p.exists():
        return "<p>(run <code>mergesort test</code> to include the correctness results)</p>"
    text = p.read_text()
    body = "".join(f"<tr><td>{m[0]}</td><td>{m[1]}</td><td>{m[2]}</td></tr>"
                   for m in re.findall(r"^\s+(\S+)\s*:\s*(\d+)\s+passed, (\d+) failed", text, re.M))
    total = next((l for l in text.splitlines() if l.startswith("TOTAL")), "")
    return (f"<table class=small><tr><th>Input type</th><th>Passed</th><th>Failed</th></tr>{body}</table>"
            f"<p style='text-align:center;font-size:9pt'>{total}</p>")


def code(func, max_lines=40):
    """Source of one function from mergesort.cpp (brace matching)."""
    lines = (ROOT / "mergesort.cpp").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("void " + func + "("))
    depth, j = 0, i
    for j in range(i, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth == 0 and "{" in "".join(lines[i:j + 1]):
            break
    snippet = "\n".join(lines[i:min(j + 1, i + max_lines)])
    return "<pre><code>" + snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</code></pre>"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


CPU = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) or platform.processor()
GXX = (run(["g++", "--version"]).splitlines() or ["g++"])[0]

# ============================================================================
# 6. REPORT
# ============================================================================

CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; color: #1b1b1b; line-height: 1.45; }
h2 { font-size: 14pt; color: #123c69; border-bottom: 1.5px solid #123c69; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #123c69; margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin: 4pt 0 7pt; text-align: justify; }
table { border-collapse: collapse; margin: 6pt auto 10pt; font-size: 9.5pt; page-break-inside: avoid; }
table.small { font-size: 8.6pt; }
th, td { border: 1px solid #b9c3cf; padding: 3pt 7pt; text-align: center; }
th { background: #e8eef5; }
tr.seq td { background: #f6f6f3; font-weight: 600; }
figure { margin: 8pt 0 10pt; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #444; margin-top: 3pt; }
pre { background: #f5f6f8; border: 1px solid #dde1e6; border-radius: 4px; padding: 6pt 8pt; font-size: 8.2pt;
      line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
code { font-family: Consolas, 'Courier New', monospace; }
.title { text-align: center; padding-top: 70pt; page-break-after: always; }
.title .uni { font-size: 15pt; font-weight: 700; }
.title .course { font-size: 12pt; margin: 4pt 0 40pt; }
.title h1 { font-size: 23pt; margin-bottom: 8pt; }
.title .sub { font-size: 13pt; color: #333; margin-bottom: 40pt; }
.title table { font-size: 11pt; margin-top: 20pt; }
.box { background: #eef4fa; border-left: 4px solid #2a78d6; padding: 6pt 10pt; margin: 8pt 0; page-break-inside: avoid; }
.eq { text-align: center; font-family: 'Cambria Math', Cambria, serif; font-size: 11pt; margin: 6pt 0; }
ul { margin: 3pt 0 7pt; } li { margin: 2pt 0; }
.pb { page-break-before: always; }
"""

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PDC Project Report - Parallel Merge Sort</title>
<style>{CSS}</style></head><body>

<div class="title">
  <div class="uni">FAST NUCES KARACHI</div>
  <div class="course">Parallel and Distributed Computing (PDC) &mdash; Semester Project Report</div>
  <h1>Parallel Merge Sort using OpenMP Tasks</h1>
  <div class="sub">2-Way vs 4-Way Merge Sort on Large Arrays</div>
  <div>Project #20</div>
  <table>
    <tr><th>Group Member</th><th>Roll Number</th></tr>
    <tr><td>Muhammad Hassan Khan</td><td>23K-0565</td></tr>
    <tr><td>Muhammad Uzair Uddin</td><td>23K-0665</td></tr>
    <tr><td>Murtaza</td><td>23K-0589</td></tr>
  </table>
  <p style="text-align:center;margin-top:30pt">Source code: <b>{GITHUB_URL}</b><br>{date.today().strftime('%d %B %Y')}</p>
</div>

<h2>Abstract</h2>
<p>We implement merge sort sequentially and in parallel with OpenMP tasks, in the two variants requested in the
proposal review: the classic <b>2-way</b> merge sort and a <b>4-way</b> merge sort that splits every range into four
quarters. Both are tested on large arrays of 32-bit integers of up to <b>{fmt_n(nmax)} elements</b> with
1&ndash;{pmax} threads. On a {CPU} (4 cores / 8 hardware threads) the best parallel run sorts {fmt_n(nmax)} integers in
<b>{best_par['time']:.2f} s</b> ({best_par['algo']}, {best_par['threads']} threads) versus {best_seq:.2f} s for the best
sequential version, a speedup of <b>{best_seq / best_par['time']:.2f}&times;</b>. A simple work/span model based on
Amdahl's law predicts the parallel times from the sequential time alone, with a mean error of <b>{mape:.1f}%</b>, and
explains why the speedup levels off. Every output was checked against <code>std::sort</code>.</p>

<h2>1. Introduction and Objectives</h2>
<p>Merge sort is a divide-and-conquer algorithm: the sub-arrays at each level are independent and can be sorted at
the same time, which maps directly onto OpenMP tasks. The proposal was <b>approved with changes</b>; this report
addresses all of them:</p>
<div class="box"><b>Instructor feedback &rarr; what we did</b>
<ul>
<li><b>2-way merge sort</b> &mdash; sequential baseline and OpenMP-task version.</li>
<li><b>4-way merge sort should be there</b> &mdash; sequential and OpenMP-task 4-way version, compared with 2-way at
every size and thread count.</li>
<li><b>Large array size</b> &mdash; arrays from 1M up to <b>{fmt_n(nmax)}</b> integers.</li>
<li><b>Accurate modelling</b> &mdash; a work/span analysis and an Amdahl's-law model that predicts the parallel run
times, checked against every measurement (Section 6).</li>
</ul></div>
<p>Objectives from the proposal: (1) implement and validate sequential merge sort; (2) parallelise it with OpenMP tasks;
(3) merge correctly; (4) evaluate on large arrays with different thread counts; (5) report time, speedup and efficiency
with tables and graphs; (6) discuss task overhead, the cutoff threshold and diminishing returns.</p>

<h2>2. Background</h2>
<h3>2.1 2-way and 4-way merge sort</h3>
<p>A k-way merge sort splits the array into k parts, sorts each part recursively and merges the k sorted parts. The
recursion has log<sub>k</sub> n levels and every level merges all n elements once, so a 4-way sort has <b>half as many
levels</b> as a 2-way sort ({math.log2(nmax):.0f} vs {math.log(nmax, 4):.0f} for n = {fmt_n(nmax)}). In exchange, each
4-way merge step must find the smallest of four values (3 comparisons) instead of two (1 comparison).</p>
<h3>2.2 OpenMP tasks</h3>
<p><code>#pragma omp task</code> creates a unit of work that any thread of the team may run, and
<code>#pragma omp taskwait</code> waits for the tasks created by the current task. A recursive algorithm can therefore
create parallel work without knowing how many threads there are. Creating a task costs time, so small sub-arrays are
sorted sequentially without creating more tasks: this is the <b>task cutoff</b>.</p>

<h2>3. Implementation</h2>
<p>All code is in one C++ file, <code>mergesort.cpp</code> (compiled with <code>g++ -O3 -fopenmp</code>). It contains
the two merge functions, the sequential and parallel sorts, and three modes: <code>test</code> (correctness),
<code>bench</code> (all experiments, results saved to <code>results/</code>) and a default demo mode. The report is
generated from the results by <code>report.py</code>.</p>
<h3>3.1 Parallel 2-way merge sort</h3>
<p>The left half is created as a task and the current thread sorts the right half itself, which saves one task per
call. After <code>taskwait</code> both halves are sorted and are merged:</p>
{code('parMergeSort2Way')}
<h3>3.2 Parallel 4-way merge sort</h3>
<p>Three quarters become tasks and the current thread sorts the fourth:</p>
{code('parMergeSort4Way')}
<p>The 4-way merge takes the smallest of the four current heads at each step:</p>
{code('merge4Way')}
<h3>3.3 Correctness testing</h3>
<p><code>mergesort test</code> compares every version element by element with <code>std::sort</code> on 17 sizes
(0, 1, 2, small odd sizes, up to 1,000,000) &times; 5 input types (random including <code>INT_MIN</code>/<code>INT_MAX</code>,
sorted, reversed, few distinct values, all equal), with 1, 2, 4 and 8 threads and cutoffs 8 and {cutoff:,}. All
tests pass (Appendix A). In the benchmarks, every timed run is also compared with <code>std::sort</code>.</p>

<h2>4. Experimental Setup</h2>
<table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>CPU</td><td>{CPU}: 4 physical cores, 8 hardware threads (Hyper-Threading)</td></tr>
<tr><td>Memory</td><td>16 GB DDR4</td></tr>
<tr><td>OS / compiler</td><td>Windows 11, {GXX}, <code>-O3 -fopenmp</code></td></tr>
<tr><td>Data</td><td>random 32-bit integers (fixed seed); input reset before every run (not timed)</td></tr>
<tr><td>Array sizes</td><td>{', '.join(fmt_n(n) for n in sizes)}</td></tr>
<tr><td>Threads</td><td>{', '.join(map(str, threads))} (plus sequential baselines)</td></tr>
<tr><td>Task cutoff</td><td>{cutoff:,} elements</td></tr>
<tr><td>Repetitions</td><td>3 per configuration, <b>median</b> reported; timer <code>omp_get_wtime()</code></td></tr>
<tr><td>Metrics</td><td>Speedup S<sub>p</sub> = T<sub>seq</sub> / T<sub>p</sub> (sequential version of the same variant),
Efficiency E<sub>p</sub> = S<sub>p</sub> / p</td></tr>
</table>
<p>The machine is a laptop: the CPU clock changes with temperature and background programs were running, which adds a
few percent of noise.</p>

<h2>5. Results</h2>
<h3>5.1 Largest array: n = {fmt_n(nmax)}</h3>
{results_table(nmax)}
{img(fig_time_vs_threads(), '70%', f'Figure 1. Execution time for n = {fmt_n(nmax)}: measured (solid), model (dotted, Section 6), sequential (dashed).')}
<h3>5.2 Speedup and efficiency for all sizes</h3>
{img(fig_per_size('speedup'), '100%', 'Figure 2. Speedup vs threads. The grey bar marks the 4 physical cores.')}
{img(fig_per_size('eff'), '100%', 'Figure 3. Parallel efficiency.')}
{all_sizes_table()}
<p style="text-align:center;font-size:9pt">Table: median times (s) for all sizes.</p>
{img(fig_time_vs_n(), '66%', 'Figure 4. Time grows slightly faster than linearly with n (n log n).')}
<h3>5.3 Observations</h3>
<ul>
<li><b>Sequential:</b> for n = {fmt_n(nmax)}, 2-way takes {s2['time']:.2f} s and 4-way {s4['time']:.2f} s, so
{seq_faster} is faster by {seq_gap:.0f}% (4-way was faster at {seq_wins4} of {len(sizes)} sizes). 4-way has half as many
levels, but each merge step makes 3 comparisons instead of 1.</li>
<li><b>Parallel:</b> the best 2-way speedup is {best2['speedup']:.2f}&times; (p = {best2['threads']}) and the best 4-way
speedup is {best4['speedup']:.2f}&times; (p = {best4['threads']}). 4-way was faster in <b>{wins4} of {len(par_cfgs)}</b>
parallel runs; at {pmax} threads and n = {fmt_n(nmax)} it takes {p4['time']:.2f} s against {p2['time']:.2f} s.</li>
<li><b>Diminishing returns:</b> speedup rises quickly up to 4 threads (one per physical core) and only a little from 4 to
{pmax}. Threads 5&ndash;{pmax} are Hyper-Threads that share a core and its memory bandwidth.</li>
<li><b>Small arrays</b> (1M) take only tens of milliseconds, so overheads matter more and speedup is lower.</li>
</ul>

<h2 class="pb">6. Performance Modelling</h2>
<h3>6.1 Work and span</h3>
<p>The work is W = &Theta;(n log n) for both variants. Each merge is done by one thread, so the longest chain of
dependent work (the span) is the sequence of merges from one leaf up to the root. With c the time to merge one
element:</p>
<p class="eq">Span<sub>2-way</sub> &asymp; c(n + n/2 + n/4 + &hellip;) = 2cn &nbsp;&nbsp;&nbsp;
Span<sub>4-way</sub> &asymp; c(n + n/4 + n/16 + &hellip;) = (4/3)cn</p>
<p>The final merge alone processes all n elements on a single thread, so the parallelism W/Span = &Theta;(log n) is
small. This serial part limits the speedup (Amdahl's law).</p>
<h3>6.2 The model</h3>
<p>The recursion has D = log<sub>k</sub> n levels and each level merges all n elements, so one level takes about
L = T<sub>seq</sub>/D. The span is then the series above, and the remaining work is shared by the cores:</p>
<p class="eq">T<sub>span</sub> = L &middot; k/(k&minus;1) &nbsp;&nbsp;&nbsp;
T<sub>p</sub> = T<sub>span</sub> + (T<sub>seq</sub> &minus; T<sub>span</sub>) / min(p, 4)</p>
<p>We use min(p, 4) because the CPU has 4 physical cores. The model only needs the measured <b>sequential</b> time, so
comparing it with the parallel runs is a fair test. For n = {fmt_n(nmax)} the serial fraction is
f = {100 * f2:.0f}% (2-way) and {100 * f4:.0f}% (4-way): even with unlimited cores the speedup could not exceed
1/f &asymp; {1 / f2:.0f}&times; and {1 / f4:.0f}&times;, and with 4 cores the model allows at most
{1 / (f2 + (1 - f2) / CORES):.1f}&times; and {1 / (f4 + (1 - f4) / CORES):.1f}&times;.</p>
<h3>6.3 Model accuracy</h3>
{img(fig_model_accuracy(), '52%', f'Figure 5. Predicted vs measured time for all {len(checked)} runs with 2 or more threads (at least 0.1 s).')}
<p>Mean error: <b>{mape:.1f}%</b>; {100 * within15:.0f}% of predictions are within &plusmn;15%. By case: 2-way
{mape_of('par2', False):.1f}% below 4 threads and {mape_of('par2', True):.1f}% at 4 or more; 4-way
{mape_of('par4', False):.1f}% and {mape_of('par4', True):.1f}%. Values for n = {fmt_n(nmax)}:</p>
{model_table()}
<p>The model assumes that no thread is ever idle. In the real program a thread that finishes its part early waits in
<code>taskwait</code> for its sibling tasks, and threads 5&ndash;8 share cores with threads 1&ndash;4, so the measured
times are usually somewhat longer than predicted at 4 or more threads.</p>

<h2>7. Task Cutoff Analysis</h2>
{img(fig_cutoff(), '62%', f'Figure 6. Run time vs task cutoff, n = {fmt_n(cut[0]["n"])}, {cut[0]["threads"]} threads.')}
<p>The best cutoff was {cut_best[2]['cutoff']:,} for 2-way ({cut_best[2]['time_s']:.3f} s) and {cut_best[4]['cutoff']:,}
for 4-way ({cut_best[4]['time_s']:.3f} s). With the smallest cutoff ({cut_small[2]['cutoff']} elements) millions of
tiny tasks are created and the time rises to {cut_small[2]['time_s']:.2f} s (2-way) and {cut_small[4]['time_s']:.2f} s
(4-way): creating and scheduling the tasks costs more than the work inside them. Every cutoff from
{near_opt[2][0]:,} to {near_opt[2][1]:,} (2-way) and {near_opt[4][0]:,} to {near_opt[4][1]:,} (4-way) is within 10% of
the best, so the exact value does not matter much once it is large enough. Even a 4M cutoff still leaves about
50M/4M &asymp; 12 leaf tasks, enough for {pmax} threads; only a cutoff close to n/p would leave threads idle.
Our default of {cutoff:,} lies in the good range.</p>

<h2>8. Discussion</h2>
<ul>
<li><b>2-way vs 4-way.</b> 4-way has half as many levels and a shorter span, and it creates parallel work faster (4
independent parts after one split instead of 2). Its merge does more comparisons per element. Which effect wins
depends on the case: here 4-way was faster in {wins4} of {len(par_cfgs)} parallel runs.</li>
<li><b>Why the speedup is far below p.</b> (1) The top merges run on one thread (the span); (2) there are only 4
physical cores; (3) all cores share the same memory bandwidth; (4) a laptop CPU lowers its clock when all cores are
busy.</li>
<li><b>Task overhead</b> only matters when the cutoff is very small; with a sensible cutoff, the serial merges limit
the speedup, not the tasks.</li>
<li><b>Possible improvement.</b> A parallel merge (splitting the merge itself between threads) would remove the serial
bottleneck; it is beyond the scope of the proposal.</li>
</ul>

<h2>9. Conclusion</h2>
<p>We implemented and validated sequential and OpenMP task-parallel 2-way and 4-way merge sorts and measured them on
arrays of up to {fmt_n(nmax)} integers with up to {pmax} threads; every output was correct. The best configuration was
{best_seq / best_par['time']:.2f}&times; faster than the best sequential sort on a 4-core/8-thread CPU. A simple
work/span model predicted the parallel times within {mape:.1f}% on average and explains the diminishing returns: the
serial top-level merges and the 4 physical cores limit the speedup, while task overhead matters only for very small
cutoffs.</p>

<h2>10. Division of Work</h2>
<table>
<tr><th>Member</th><th>Responsibilities</th></tr>
<tr><td>Muhammad Hassan Khan (23K-0565)</td><td>Sequential 2-way and 4-way merge sort, correctness testing</td></tr>
<tr><td>Muhammad Uzair Uddin (23K-0665)</td><td>OpenMP task implementation, cutoff-threshold experiments</td></tr>
<tr><td>Murtaza (23K-0589)</td><td>Benchmarking, performance model, graphs and report</td></tr>
</table>

<h2 class="pb">Appendix A: Detailed Results</h2>
<h3>A.1 Correctness tests</h3>
{tests_table()}
<h3>A.2 Task cutoff sweep (n = {fmt_n(cut[0]['n'])}, {cut[0]['threads']} threads, median of 3 runs)</h3>
{cutoff_table()}

<h2>Appendix B: Reproducing the Results</h2>
<pre><code>git clone {GITHUB_URL}
cd parallel-merge-sort-openmp
g++ -O3 -fopenmp mergesort.cpp -o mergesort    # add -static on MSYS2 if linking fails
./mergesort test        # correctness tests
./mergesort             # quick demo: 50M integers, 8 threads
./mergesort bench       # all experiments -> results/ (about 25 min)
python report.py        # this PDF</code></pre>
</body></html>
"""

# ============================================================================
# 7. HTML -> PDF with headless Edge / Chrome
# ============================================================================

html_path = RES / "report.html"
html_path.write_text(html, encoding="utf-8")

browsers = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("msedge") or ""]
exe = next((b for b in browsers if b and Path(b).exists()), None)
if exe is None:
    sys.exit(f"No Edge/Chrome found: open {html_path} in a browser and print it to PDF.")

if PDF.exists():
    PDF.unlink()
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                    f"--print-to-pdf={PDF}", html_path.as_uri()], capture_output=True, timeout=180)
    for _ in range(60):  # the browser may finish writing after it returns
        if PDF.exists() and PDF.stat().st_size > 0:
            time.sleep(1)
            break
        time.sleep(0.5)
print(f"Report written: {PDF.name} ({PDF.stat().st_size // 1024} KB)" if PDF.exists() else "PDF generation failed")
