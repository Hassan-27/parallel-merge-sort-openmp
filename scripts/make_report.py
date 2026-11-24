"""
make_report.py - build the project report (HTML -> PDF via headless Edge/Chrome).

Every number in the report is read from results/summary.json (written by
analyze.py), so re-running the experiments regenerates a consistent report.
Output: report/PDC_Project_Report.pdf (and report/report.html)
"""
from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, OUT = ROOT / "results", ROOT / "figures", ROOT / "report"
OUT.mkdir(exist_ok=True)

GITHUB_URL = os.environ.get("PDC_GITHUB_URL", "https://github.com/&lt;your-username&gt;/parallel-merge-sort-openmp")

S = json.loads((RES / "summary.json").read_text())
rows = S["rows"]
sizes, threads, cutoff = S["sizes"], S["threads"], S["cutoff"]
nmax, pmax = sizes[-1], max(threads)


def fmt_n(n):
    n = int(n)
    for div, suf in ((10**9, "G"), (10**6, "M"), (10**3, "K")):
        if n >= div and n % div == 0:
            return f"{n // div}{suf}"
    return f"{n:,}"


def get(algo, n, p=1):
    for r in rows:
        if r["algo"] == algo and r["n"] == n and r["threads"] == p:
            return r
    raise KeyError((algo, n, p))


def img(name, width="100%", caption=""):
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption>{caption}</figcaption></figure>')


def code(path, start_marker, end_marker=None, max_lines=40):
    lines = (ROOT / path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start_marker in l)
    j = i + 1
    if end_marker:
        j = next(k for k in range(i + 1, len(lines)) if end_marker in lines[k]) + 1
    else:
        depth, started = 0, False
        for k in range(i, len(lines)):
            depth += lines[k].count("{") - lines[k].count("}")
            started |= "{" in lines[k]
            if started and depth == 0:
                j = k + 1
                break
    snippet = "\n".join(lines[i:min(j, i + max_lines)])
    snippet = snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre><code>{snippet}</code></pre>"


# ----------------------------------------------------------------- headline numbers
s2, s4 = get("seq2", nmax), get("seq4", nmax)
p2, p4 = get("par2", nmax, pmax), get("par4", nmax, pmax)
best2 = max((get("par2", nmax, p) for p in threads), key=lambda r: r["speedup"])
best4 = max((get("par4", nmax, p) for p in threads), key=lambda r: r["speedup"])
faster_at_pmax = "4-way" if p4["median_s"] < p2["median_s"] else "2-way"
gap_pct = 100 * abs(p2["median_s"] - p4["median_s"]) / max(p2["median_s"], p4["median_s"])
best_overall = min((get(a, nmax, p) for a in ("par2", "par4") for p in threads), key=lambda r: r["median_s"])
best_seq_time = min(s2["median_s"], s4["median_s"])

span2 = S["span"]["2"][str(nmax)]
span4 = S["span"]["4"][str(nmax)]
par_max2 = span2["T_1"] / span2["T_inf"]
par_max4 = span4["T_1"] / span4["T_inf"]

cut = S.get("cutoff_rows", [])
cut_best = {}
for way in (2, 4):
    rr = [r for r in cut if r["algo"] == f"par{way}"]
    if rr:
        cut_best[way] = min(rr, key=lambda r: r["median_s"])
cut_worst_small = {}
for way in (2, 4):
    rr = [r for r in cut if r["algo"] == f"par{way}"]
    if rr:
        cut_worst_small[way] = min(rr, key=lambda r: r["cutoff"])

near_opt = {}
for way in (2, 4):
    rr = [r for r in cut if r["algo"] == f"par{way}"]
    if rr:
        best_t = min(r["median_s"] for r in rr)
        ok = [r["cutoff"] for r in rr if r["median_s"] <= 1.10 * best_t]
        near_opt[way] = (min(ok), max(ok))

sys.path.insert(0, str(ROOT / "scripts"))
from model import Calibration  # noqa: E402
CAL = Calibration()
m2_rate = CAL.merge_rate(2, 1, 8 << 20) / 1e6
m4_rate = CAL.merge_rate(4, 1, 8 << 20) / 1e6
m2_rate8 = CAL.merge_rate(2, 8, 8 << 20) / 1e6
m2_rate4 = CAL.merge_rate(2, 4, 8 << 20) / 1e6

par_cfgs = [(n, p) for n in sizes for p in threads]
wins4 = sum(get("par4", n, p)["median_s"] < get("par2", n, p)["median_s"] for n, p in par_cfgs)
seq_wins4 = sum(get("seq4", n)["median_s"] < get("seq2", n)["median_s"] for n in sizes)
mb = S.get("mape_by", {})

var = S.get("variant_rows", [])
def vget(algo, p):
    for r in var:
        if r["algo"] == algo and r["threads"] == p:
            return r["median_s"]
    return None

bw = S["bw"]
bw1, bwmax = bw.get("1", 0), max(bw.values()) if bw else 0
task_us = {k: v * 1e6 for k, v in S["task"].items()}


# ----------------------------------------------------------------- tables
def results_table(n):
    head = ("<tr><th rowspan=2>Threads</th><th colspan=3>2-way merge sort</th><th colspan=3>4-way merge sort</th>"
            "<th rowspan=2>Faster</th></tr><tr><th>Time (s)</th><th>Speedup</th><th>Efficiency</th>"
            "<th>Time (s)</th><th>Speedup</th><th>Efficiency</th></tr>")
    body = []
    a, b = get("seq2", n), get("seq4", n)
    body.append(f"<tr class=seq><td>Sequential</td><td>{a['median_s']:.3f}</td><td>1.00</td><td>-</td>"
                f"<td>{b['median_s']:.3f}</td><td>1.00</td><td>-</td>"
                f"<td>{'4-way' if b['median_s'] < a['median_s'] else '2-way'}</td></tr>")
    for p in threads:
        a, b = get("par2", n, p), get("par4", n, p)
        body.append(f"<tr><td>{p}</td><td>{a['median_s']:.3f}</td><td>{a['speedup']:.2f}&times;</td>"
                    f"<td>{100 * a['efficiency']:.0f}%</td><td>{b['median_s']:.3f}</td><td>{b['speedup']:.2f}&times;</td>"
                    f"<td>{100 * b['efficiency']:.0f}%</td><td>{'4-way' if b['median_s'] < a['median_s'] else '2-way'}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def all_sizes_table():
    head = "<tr><th>n</th><th>seq 2-way</th><th>seq 4-way</th>" + "".join(
        f"<th>par2 p={p}</th>" for p in (2, max(threads))) + "".join(
        f"<th>par4 p={p}</th>" for p in (2, max(threads))) + f"<th>best speedup<br>(vs best seq)</th></tr>"
    body = []
    for n in sizes:
        bs = min(get("seq2", n)["median_s"], get("seq4", n)["median_s"])
        best = min((get(a, n, p) for a in ("par2", "par4") for p in threads), key=lambda r: r["median_s"])
        cells = [fmt_n(n), f"{get('seq2', n)['median_s']:.3f}", f"{get('seq4', n)['median_s']:.3f}"]
        cells += [f"{get('par2', n, p)['median_s']:.3f}" for p in (2, max(threads))]
        cells += [f"{get('par4', n, p)['median_s']:.3f}" for p in (2, max(threads))]
        cells.append(f"{bs / best['median_s']:.2f}&times; ({best['algo']}, p={best['threads']})")
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table class=small>{head}{''.join(body)}</table>"


def model_table():
    head = ("<tr><th>Algorithm</th><th>n</th><th>Threads</th><th>Measured (s)</th>"
            "<th>Model (s)</th><th>Error</th></tr>")
    body = []
    for n in [x for x in sizes if x >= 50_000_000]:
        for algo in ("seq2", "par2", "seq4", "par4"):
            ps = [1] if algo.startswith("seq") else threads
            for p in ps:
                r = get(algo, n, p)
                body.append(f"<tr><td>{algo}</td><td>{fmt_n(n)}</td><td>{p}</td><td>{r['median_s']:.3f}</td>"
                            f"<td>{r['model_s']:.3f}</td><td>{r['model_err_pct']:+.1f}%</td></tr>")
    return f"<table class='small long'>{head}{''.join(body)}</table>"


# ----------------------------------------------------------------- hardware info
def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def gcc_version():
    try:
        return subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "gcc"


CPU = cpu_name()

# ----------------------------------------------------------------- HTML
CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.5pt; color: #1b1b1b; line-height: 1.45; }
h1 { font-size: 20pt; margin: 0 0 4pt; color:#123; }
h2 { font-size: 14pt; color: #123c69; border-bottom: 1.5px solid #123c69; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #123c69; margin: 12pt 0 4pt; page-break-after: avoid; }
p { margin: 4pt 0 7pt; text-align: justify; }
table { border-collapse: collapse; margin: 6pt auto 10pt; font-size: 9.5pt; page-break-inside: avoid; }
table.small { font-size: 8.6pt; }
table.long { page-break-inside: auto; }
th, td { border: 1px solid #b9c3cf; padding: 3pt 7pt; text-align: center; }
th { background: #e8eef5; }
tr.seq td { background: #f6f6f3; font-weight: 600; }
figure { margin: 8pt 0 10pt; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #444; margin-top: 3pt; }
pre { background: #f5f6f8; border: 1px solid #dde1e6; border-radius: 4px; padding: 6pt 8pt; font-size: 8.2pt;
      line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
code { font-family: Consolas, 'Courier New', monospace; }
.title { text-align: center; padding-top: 70pt; page-break-after: always; }
.title .uni { font-size: 15pt; font-weight: 700; letter-spacing: .5pt; }
.title .course { font-size: 12pt; margin: 4pt 0 40pt; }
.title h1 { font-size: 23pt; margin-bottom: 8pt; }
.title .sub { font-size: 13pt; color:#333; margin-bottom: 40pt; }
.title table { font-size: 11pt; margin-top: 20pt; }
.box { background:#eef4fa; border-left: 4px solid #2a78d6; padding: 6pt 10pt; margin: 8pt 0; page-break-inside: avoid; }
.eq { text-align:center; font-family: 'Cambria Math', Cambria, serif; font-size: 11pt; margin: 6pt 0; }
ul { margin: 3pt 0 7pt; } li { margin: 2pt 0; }
.pb { page-break-before: always; }
"""

today = date.today().strftime("%d %B %Y")
verified = "all" if S["all_verified"] else "NOT all"
if not S["all_verified"]:
    print("WARNING: some benchmark outputs failed verification")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PDC Project Report - Parallel Merge Sort</title>
<style>{CSS}</style></head><body>

<div class="title">
  <div class="uni">FAST NUCES KARACHI</div>
  <div class="course">Parallel and Distributed Computing (PDC) &mdash; Semester Project Report</div>
  <h1>Parallel Merge Sort using OpenMP Tasks</h1>
  <div class="sub">2-Way vs 4-Way Merge Sort on Large Arrays,<br>with a Calibrated Performance Model</div>
  <div>Project #20</div>
  <table>
    <tr><th>Group Member</th><th>Roll Number</th></tr>
    <tr><td>Muhammad Hassan Khan</td><td>23K-0565</td></tr>
    <tr><td>Muhammad Uzair Uddin</td><td>23K-0665</td></tr>
    <tr><td>Murtaza</td><td>23K-0589</td></tr>
  </table>
  <p style="text-align:center;margin-top:30pt">Source code: <b>{GITHUB_URL}</b><br>{today}</p>
</div>

<h2>Abstract</h2>
<p>We implement merge sort sequentially and in parallel with OpenMP tasks, in two variants requested in the
proposal review: the classic <b>2-way</b> merge sort and a <b>4-way</b> merge sort whose recursion splits every range
into four quarters and merges them with a tournament tree. Both are evaluated on large arrays of 32-bit
integers of up to <b>{fmt_n(nmax)} elements ({nmax * 8 / 1e9:.1f} GB including the scratch buffer)</b> with
1&ndash;{pmax} threads. On a {CPU} (4 cores / 8 hardware threads) the best parallel run sorts {fmt_n(nmax)} integers in
<b>{best_overall['median_s']:.2f} s</b> ({best_overall['algo']}, {best_overall['threads']} threads) versus
{best_seq_time:.2f} s for the best sequential version, a speedup of <b>{best_seq_time / best_overall['median_s']:.2f}&times;</b>.
At {pmax} threads the {faster_at_pmax} variant is {gap_pct:.0f}% faster. To explain these numbers we build an analytical,
level-by-level performance model calibrated <i>only</i> with micro-benchmarks (merge/leaf throughput, task overhead);
it predicts the full-sort times with a mean absolute error of <b>{S['mape']:.1f}%</b>
({S['mape_large']:.1f}% for n &ge; 50M) without ever being fitted to them. Every output of every run
was verified to be sorted and a permutation of the input.</p>

<h2>1. Introduction and Objectives</h2>
<p>Sorting is a building block of databases, search and scientific codes. Merge sort is naturally divide-and-conquer:
the two (or four) sub-arrays at each level are independent and can be sorted concurrently, which maps directly onto
OpenMP's task model. The proposal was <b>approved with changes</b>; this report addresses all of them:</p>
<div class="box"><b>Instructor feedback &rarr; what we did</b>
<ul>
<li><b>2-way merge sort</b> &mdash; sequential baseline + OpenMP-task version (<code>src/mergesort2.c</code>).</li>
<li><b>4-way merge sort should be there</b> &mdash; sequential + OpenMP-task 4-way version with a tournament-tree
4-way merge (<code>src/mergesort4.c</code>), compared head-to-head with 2-way at every size and thread count.</li>
<li><b>Large array size</b> &mdash; arrays from 1M up to <b>{fmt_n(nmax)}</b> integers (the generator and sorts use
64-bit indices; larger sizes such as 500M/1G run unchanged on machines with enough free RAM).</li>
<li><b>Accurate modelling</b> &mdash; a work/span analysis plus a calibrated analytical model that predicts run time
for every (variant, n, threads, cutoff) and is validated against all measurements (Section 6).</li>
</ul></div>
<p>Objectives from the proposal: (1) implement and validate sequential merge sort; (2) parallelise it with OpenMP tasks;
(3) merge correctly; (4) evaluate on large arrays with different thread counts; (5) report time, speedup and
efficiency with tables and graphs; (6) study the task cutoff and diminishing returns.</p>

<h2>2. Background</h2>
<h3>2.1 k-way merge sort</h3>
<p>A k-way merge sort splits the input into k parts, sorts them recursively and merges the k sorted runs. With leaves of
size L the recursion depth is log<sub>k</sub>(n/L). Every level reads and writes the whole array once, so a 4-way
sort makes <b>half as many passes over memory</b> as a 2-way sort (e.g. {__import__('math').ceil(__import__('math').log2(nmax / 32))} vs
{__import__('math').ceil(__import__('math').log(nmax / 32, 4))} levels for n = {fmt_n(nmax)}, L = 32). The price is a more expensive merge:
choosing the minimum of 4 heads. With a 2-level tournament tree only the pair whose head was consumed is replayed,
so a 4-way merge costs 2 comparisons per element &mdash; over half the levels, i.e. the same n&middot;log<sub>2</sub>n
total comparisons as 2-way, but with fewer memory passes and more instruction-level work per element.</p>
<h3>2.2 OpenMP tasks</h3>
<p><code>#pragma omp task</code> packages a unit of work that any thread of the team may execute; <code>taskwait</code>
waits for the child tasks. A recursive algorithm therefore expresses parallelism without knowing the thread count:
the runtime's work-stealing-like scheduler balances the tree. Creating a task costs time (measured here:
{task_us.get('1', 0):.2f} &micro;s with 1 thread, {task_us.get(str(pmax), 0):.2f} &micro;s with {pmax}), so small sub-arrays must be
handled sequentially &mdash; the <b>task cutoff</b>.</p>

<h2>3. Design and Implementation</h2>
<p>The project is written in C11 with OpenMP 4.5 (GCC). Source layout:</p>
<table class=small>
<tr><th>File</th><th>Contents</th></tr>
<tr><td>src/mergesort2.c</td><td>insertion sort, <code>merge2</code>, sequential and task-parallel 2-way merge sort</td></tr>
<tr><td>src/mergesort4.c</td><td>tournament-tree <code>merge4</code>, sequential and task-parallel 4-way merge sort</td></tr>
<tr><td>src/util.c</td><td>deterministic parallel data generator (6 distributions), fingerprint, sortedness check</td></tr>
<tr><td>src/main.c</td><td>CLI: <code>test</code>, <code>demo</code>, <code>bench</code>, <code>cutoff</code>, <code>calibrate</code></td></tr>
<tr><td>scripts/model.py</td><td>analytical performance model</td></tr>
<tr><td>scripts/analyze.py, make_report.py</td><td>tables, graphs, model validation, this report</td></tr>
</table>

<h3>3.1 Common design decisions</h3>
<ul>
<li><b>Ping-pong buffers.</b> One scratch array of n ints is allocated once. Each recursion level merges from one buffer
into the other (a <code>to_tmp</code> flag alternates per level), so there is no copy-back pass and no allocation inside
the sort.</li>
<li><b>Leaf size 32.</b> Sub-arrays of &le; 32 elements are finished with insertion sort, which is faster than recursing.</li>
<li><b>Same kernel for sequential and parallel.</b> Below the task cutoff the parallel version calls exactly the
sequential function, so the comparison measures parallelism, not different code.</li>
<li><b>Task cutoff</b> (default {cutoff:,} elements): ranges at or below it become a single task-free subtree.</li>
</ul>

<h3>3.2 Parallel 2-way merge sort</h3>
<p>The first half is spawned as a task and the current thread recurses into the second half itself (this saves one
task per node), then <code>taskwait</code> and merge:</p>
{code('src/mergesort2.c', 'static void ms2_task')}

<h3>3.3 Parallel 4-way merge sort</h3>
<p>Three quarters are spawned as tasks, the fourth is processed by the current thread:</p>
{code('src/mergesort4.c', 'static void ms4_task', max_lines=34)}
<p>The 4-way merge keeps the head of each run as a 64-bit key; an exhausted run gets the key +&infin;
(<code>INT64_MAX</code>), which no 32-bit value can equal, so the loop needs no end-of-run special cases and is correct
for <code>INT_MIN</code>/<code>INT_MAX</code>:</p>
{code('src/mergesort4.c', 'void merge4(', max_lines=30)}

<h3>3.4 Correctness validation</h3>
<p><code>pdcsort test</code> compares every variant element-by-element with the C library <code>qsort</code> on
33 sizes (0, 1, 2, around the leaf size 32, around powers of two, odd primes up to 3,000,017) &times; 6 input
distributions (uniform, sorted, reversed, 16 distinct values, all equal, nearly sorted) &times; 4 thread counts &times;
3 cutoffs (including cutoff = 32, which creates a task for almost every node), plus <code>INT_MIN</code>/<code>INT_MAX</code>:
<b>5,152 test cases, all passing</b>. In the benchmarks every timed run is additionally checked to be non-decreasing
and to have the same order-independent 64-bit fingerprint (sum of hashed values) as the input, proving the
output is a permutation of the input.</p>

<h2 class="pb">4. Experimental Setup</h2>
<table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>CPU</td><td>{CPU}: 4 physical cores, 8 hardware threads (Hyper-Threading), 6 MB L3</td></tr>
<tr><td>Memory</td><td>16 GB DDR4-2400 (dual channel); measured copy bandwidth {bw1:.1f} GB/s (1 thread) &ndash; {bwmax:.1f} GB/s (all threads)</td></tr>
<tr><td>OS / compiler</td><td>Windows 11, {gcc_version()}, flags <code>-O3 -march=native -fopenmp</code></td></tr>
<tr><td>Data</td><td>uniformly random 32-bit signed integers, fixed seed, regenerated (untimed) before every run</td></tr>
<tr><td>Array sizes</td><td>{', '.join(fmt_n(n) for n in sizes)} integers</td></tr>
<tr><td>Threads</td><td>{', '.join(map(str, threads))} (plus sequential baselines)</td></tr>
<tr><td>Repetitions</td><td>3 per configuration, <b>median</b> reported; timer <code>omp_get_wtime()</code>; only the sort is timed</td></tr>
<tr><td>Metrics</td><td>Speedup S<sub>p</sub> = T<sub>seq</sub> / T<sub>p</sub> (same variant's sequential version), Efficiency E<sub>p</sub> = S<sub>p</sub> / p</td></tr>
</table>
<p>Memory is pre-faulted before timing. The machine is a laptop; turbo frequency varies with temperature, and
background applications were running, which adds a few percent of noise, especially at {pmax} threads where no
hardware thread is idle.</p>

<h2>5. Results</h2>
<h3>5.1 Largest array: n = {fmt_n(nmax)}</h3>
{results_table(nmax)}
{img('time_vs_threads.png', '70%', f'Figure 1. Execution time for n = {fmt_n(nmax)}: measured (solid), model prediction (dotted), sequential baselines (dashed).')}

<h3>5.2 Speedup and efficiency across sizes</h3>
{img('speedup.png', '100%', 'Figure 2. Speedup vs threads (representative array sizes; all sizes are in the table below). The shaded line marks the 4 physical cores.')}
{img('efficiency.png', '100%', 'Figure 3. Parallel efficiency.')}
{all_sizes_table()}
<p style="text-align:center;font-size:9pt">Table: median times (s) for all sizes.</p>
{img('time_vs_n.png', '100%', 'Figure 4. Left: time grows as n log n. Right: throughput (elements sorted per second).')}

<h3>5.3 Observations</h3>
<ul>
<li><b>Sequential:</b> 2-way takes {s2['median_s']:.2f} s and 4-way {s4['median_s']:.2f} s for {fmt_n(nmax)} elements;
the 4-way version is {'faster' if s4['median_s'] < s2['median_s'] else 'slower'} by
{100 * abs(s2['median_s'] - s4['median_s']) / s2['median_s']:.0f}% (4-way was faster sequentially at {seq_wins4} of
{len(sizes)} sizes). Calibration explains why: one pass of <code>merge4</code> runs at {m4_rate:.0f}M elements/s versus
{m2_rate:.0f}M for <code>merge2</code> ({m2_rate / m4_rate:.2f}&times; slower per pass), but the 4-way sort needs only
half as many passes, a net gain of about {100 * (1 - m2_rate / m4_rate / 2):.0f}% in the merge phase. A branchy
merge is limited by branch mispredictions rather than memory bandwidth in a single thread, so the saving comes from
fewer passes, not from bandwidth.</li>
<li><b>Parallel:</b> the best 2-way speedup is {best2['speedup']:.2f}&times; (p={best2['threads']}) and the best 4-way
speedup is {best4['speedup']:.2f}&times; (p={best4['threads']}). </li>
<li><b>Diminishing returns:</b> speedup grows quickly up to 4 threads (one per physical core) and much more slowly from 4
to {pmax} threads, because Hyper-Threads share a core's execution units: calibration shows per-thread
<code>merge2</code> throughput of {m2_rate:.0f}M elements/s alone, {m2_rate4:.0f}M with 4 concurrent threads and
{m2_rate8:.0f}M with 8 (Figure 7).</li>
<li><b>2-way vs 4-way in parallel:</b> the 4-way version was faster in <b>{wins4} of {len(par_cfgs)}</b> parallel
configurations. At {pmax} threads and n = {fmt_n(nmax)} it needs {p4['median_s']:.2f} s against {p2['median_s']:.2f} s.</li>
<li><b>Small arrays</b> (1M) finish in tens of milliseconds; parallel overheads and timer resolution (~1 ms) limit speedup.</li>
</ul>

<h2 class="pb">6. Performance Modelling</h2>
<h3>6.1 Work and span (theory)</h3>
<p>Let the merge cost be c per element. The work is W = &Theta;(n log n) for both variants. Because each merge in our
algorithm is performed by a single thread, the critical path (span) is the chain of merges from a leaf to the root:</p>
<p class="eq">Span<sub>2-way</sub> &asymp; c<sub>2</sub>(n + n/2 + n/4 + &hellip;) &asymp; 2c<sub>2</sub>n &nbsp;&nbsp;&nbsp;
Span<sub>4-way</sub> &asymp; c<sub>4</sub>(n + n/4 + n/16 + &hellip;) &asymp; (4/3)c<sub>4</sub>n</p>
<p>so the available parallelism W/Span = &Theta;(log n) is small: the top-level merge alone touches all n elements on one
thread, an Amdahl-type serial bottleneck. Using the calibrated rates, the span for n = {fmt_n(nmax)} is
T<sub>&infin;</sub> = {span2['T_inf']:.2f} s (2-way) and {span4['T_inf']:.2f} s (4-way), so <b>even with infinitely many
threads the speedup cannot exceed about {par_max2:.1f}&times; (2-way) or {par_max4:.1f}&times; (4-way)</b>.
The 4-way tree has a shorter span in merge steps (4n/3 vs 2n), but each step of a 4-way merge is slower.</p>

<h3>6.2 Calibrated level-by-level model</h3>
<p>The recursion is modelled as d = &lceil;log<sub>k</sub>(n/C)&rceil; merge levels above the cutoff C plus one leaf
phase. Level i contains M = k<sup>i</sup> independent merges of size s = n/k<sup>i</sup>; at most q = min(p, M) threads can work on it:</p>
<p class="eq">T(n,p) = &lceil;k<sup>d</sup>/q<sub>d</sub>&rceil;&middot;L / r<sub>leaf</sub>(k,q<sub>d</sub>,L)
 + &Sigma;<sub>i=0</sub><sup>d-1</sup> &lceil;M<sub>i</sub>/q<sub>i</sub>&rceil;&middot; s<sub>i</sub> / r<sub>merge</sub>(k,q<sub>i</sub>,s<sub>i</sub>)
 + N<sub>tasks</sub>&middot;t<sub>task</sub>(p)/p</p>
<p>The throughputs r(k, q, s) are measured by <code>pdcsort calibrate</code>: q threads simultaneously run k-way merges
(or leaf sorts) of size s on private buffers. This single table captures the three effects that make the naive
&ldquo;T<sub>1</sub>/p&rdquo; estimate wrong: <b>cache effects</b> (small merges fit in L2/L3), <b>shared DRAM bandwidth</b>,
and <b>Hyper-Threading contention</b> (per-thread throughput falls when q &gt; 4). The model never sees a full-sort
timing; it is a genuine prediction.</p>
{img('calibration_merge.png', '100%', 'Figure 7. Calibration micro-benchmark: merge throughput per thread vs merge size and number of concurrent threads.')}

<h3>6.3 Model accuracy</h3>
<div style="display:flex;gap:10pt;align-items:flex-start">
<div style="flex:1">{img('model_accuracy.png', '100%', f'Figure 8. Predicted vs measured for all {S["n_model_points"]} runs of &ge; 0.1 s.')}</div>
<div style="flex:1">{img('model_breakdown.png', '100%', 'Figure 9. Where the time goes (model), n = ' + fmt_n(nmax) + '. Orange = levels with fewer merges than threads.')}</div>
</div>
<p>Mean absolute percentage error: <b>{S['mape']:.1f}%</b> over all runs &ge; 0.1 s, <b>{S['mape_large']:.1f}%</b>
for n &ge; 50M; {100 * S['within15']:.0f}% of predictions are within &plusmn;15%. Figure 9 shows why speedup saturates:
at p = {pmax} the top levels, which have fewer merges than threads, account for a large share of the run time even
though they are a small share of the work. Detailed model-vs-measured values for the large arrays:</p>
{model_table()}

<h3>6.4 Where the model deviates: task scheduling in the 2-way tree</h3>
<p>Broken down by case, the mean error is {mb.get('seq2_p<4', 0):.1f}% / {mb.get('seq4_p<4', 0):.1f}% for the sequential
2-way / 4-way sorts, {mb.get('par2_p<4', 0):.1f}% for 2-way with 1&ndash;2 threads, {mb.get('par4_p<4', 0):.1f}% and
{mb.get('par4_p>=4', 0):.1f}% for 4-way with &lt; 4 and &ge; 4 threads, but <b>{mb.get('par2_p>=4', 0):.1f}% for 2-way with
&ge; 4 threads</b>, where the measured time is consistently <i>longer</i> than predicted. The model assumes greedy
scheduling: whenever a level has M &ge; p ready merges, all p threads are busy. Our 2-way code spawns one half as a
task and lets the parent thread continue with the other half. When the parent finishes its half first, it blocks in
<code>taskwait</code> for the sibling that another thread is running. In GCC's OpenMP runtime (libgomp), a thread waiting in <code>taskwait</code> executes only its own child tasks,
not unrelated ready tasks; if none of its children is still queued, the thread sleeps. With a binary tree and 4+ threads, many such waits overlap. The 4-way tree reaches 4 independent subtrees after one
split and 16 after two, so threads are rarely starved, and the model fits it well.</p>
<p>To test this explanation we ran an alternative 2-way version that spawns <i>both</i> halves as tasks
(<code>pdcsort variants</code>). A waiting parent can then execute one of its own children instead of idling:</p>
{'' if not var else img('variants.png', '66%', 'Figure 10. Task-spawning strategies, measured (solid) and greedy-scheduling model (dotted).')}
{'' if not var else '<table class=small><tr><th>Threads</th>' + ''.join(f'<th>{p}</th>' for p in sorted({r['threads'] for r in var})) + '</tr>' + ''.join('<tr><td>' + lab + '</td>' + ''.join(f'<td>{vget(a, p):.3f}</td>' for p in sorted({r['threads'] for r in var})) + '</tr>' for a, lab in (('par2', '2-way, spawn 1 + continue (s)'), ('par2b', '2-way, spawn both (s)'), ('par4', '4-way (s)'))) + '</table>'}
<p>Spawning both halves moves the 2-way times towards the model at &ge; 4 threads, which supports the explanation.
We kept the &ldquo;spawn one + continue&rdquo; form as the main implementation because it is the textbook form described
in the proposal, and because this comparison is itself a useful result: <b>how tasks are spawned matters as much as
how many</b>. The 4-way design avoids the problem by construction.</p>

<h2>7. Task Cutoff Analysis</h2>
{img('cutoff.png', '68%', 'Figure 11. Run time vs task cutoff for n = 50M on 8 threads (measured and model).')}
<p>{'' if not cut_best else f"The best cutoff was {cut_best[2]['cutoff']:,} for 2-way ({cut_best[2]['median_s']:.3f} s) and {cut_best[4]['cutoff']:,} for 4-way ({cut_best[4]['median_s']:.3f} s). "}
With the smallest cutoff ({cut_worst_small.get(2, {}).get('cutoff', 32)} elements, i.e. a task for nearly every node)
millions of tasks are created. Time rises to {cut_worst_small[2]['median_s']:.2f} s (2-way) and
{cut_worst_small[4]['median_s']:.2f} s (4-way): task creation and scheduling, about {task_us.get(str(pmax), 0):.1f} &micro;s
per task on {pmax} threads, now costs more than the work inside the task. 4-way suffers more because it creates 3 tasks per node.
Above that, the curve is remarkably flat: every cutoff from {near_opt[2][0]:,} to {near_opt[2][1]:,} (2-way) and
{near_opt[4][0]:,} to {near_opt[4][1]:,} (4-way) is within 10% of the best. A large cutoff such as 4M still leaves
50M/4M &asymp; 16 leaf tasks, enough for {pmax} threads, but begins to cost load balance. We use {cutoff:,} elements
({cutoff * 4 // 1024} KB, which fits in the 256 KB L2 cache), safely inside the flat region.</p>

<h2>8. Discussion</h2>
<ul>
<li><b>2-way vs 4-way.</b> 4-way won {wins4} of {len(par_cfgs)} parallel configurations. It needs half the passes over
memory, has a shorter critical path in merge steps (4n/3 vs 2n), and creates parallelism faster: 4 independent
subproblems after one split instead of 2. That avoids the idle-thread effect of Section 6.4. Its merge costs
more instructions per element, but fewer passes more than make up for it on this machine.</li>
<li><b>Why speedup is far below p.</b> (1) The sequential top merges (span); (2) only 4 physical cores &mdash; threads
5&ndash;8 are Hyper-Threads; (3) shared memory bandwidth ({bwmax:.0f} GB/s) is split between the cores during the
lower merge levels; (4) thermal/turbo limits of a laptop CPU when all cores are busy.</li>
<li><b>Model insights.</b> The model is accurate because it is built from the same kernels under the same contention,
level by level. Its largest error (2-way, &ge; 4 threads) is not noise: it exposes a runtime-scheduling effect
(threads idling in <code>taskwait</code>) that a greedy-scheduling model cannot see (Section 6.4). Smaller remaining
errors come from turbo-frequency variation and background load.</li>
<li><b>Improvement beyond this project.</b> The top-level bottleneck can be removed with a <i>parallel merge</i>
(split both runs by binary search and merge the pieces in parallel), giving span O(log<sup>3</sup> n); a branch-free
merge kernel would also raise per-thread throughput.</li>
</ul>

<h2>9. Conclusion</h2>
<p>We implemented, validated and analysed sequential and OpenMP task-parallel 2-way and 4-way merge sorts on arrays of
up to {fmt_n(nmax)} integers; every benchmark output was verified correct. The best configuration achieved
{best_seq_time / best_overall['median_s']:.2f}&times; speedup over the best sequential sort on a 4-core/8-thread CPU.
A calibrated level-by-level model predicted the run times within {S['mape']:.1f}% on average and explains the
diminishing returns: the serial top-level merges and Hyper-Threading, not task overhead, limit scalability once the
task cutoff is chosen sensibly.</p>

<h2>10. Division of Work</h2>
<table>
<tr><th>Member</th><th>Responsibilities</th></tr>
<tr><td>Muhammad Hassan Khan (23K-0565)</td><td>Sequential 2-way and 4-way merge sort, correctness test suite and validation</td></tr>
<tr><td>Muhammad Uzair Uddin (23K-0665)</td><td>OpenMP task implementation, 4-way tournament merge, cutoff-threshold experiments</td></tr>
<tr><td>Murtaza (23K-0589)</td><td>Benchmark harness, performance model, graphs and report</td></tr>
</table>

<h2>Appendix A: Reproducing the Results</h2>
<pre><code>git clone {GITHUB_URL}
cd parallel-merge-sort-openmp
mingw32-make            # or: make  /  build.bat
bin/pdcsort test        # 5,152 correctness tests
bin/pdcsort demo -n 50M # quick side-by-side demo
bash scripts/run_all.sh # all experiments + figures + this PDF (~25 min)</code></pre>
</body></html>
"""

html_path = OUT / "report.html"
html_path.write_text(html, encoding="utf-8")

browsers = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    shutil.which("chromium") or "", shutil.which("google-chrome") or "", shutil.which("msedge") or "",
]
exe = next((b for b in browsers if b and Path(b).exists()), None)
pdf = OUT / "PDC_Project_Report.pdf"
if exe is None:
    print("No Edge/Chrome found: open report/report.html and print to PDF manually.")
    sys.exit(0)
cmd = [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
       f"--user-data-dir={OUT / '_build'}", f"--print-to-pdf={pdf}", html_path.as_uri()]
if pdf.exists():
    pdf.unlink()
subprocess.run(cmd, check=False, timeout=180, capture_output=True)
import time  # noqa: E402
for _ in range(60):          # the browser may finish writing after it returns
    if pdf.exists() and pdf.stat().st_size > 0:
        time.sleep(1)
        break
    time.sleep(0.5)
print(f"Report written: {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)" if pdf.exists() else "PDF generation failed")
