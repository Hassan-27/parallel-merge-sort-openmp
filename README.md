# Parallel Merge Sort using OpenMP Tasks — 2-Way vs 4-Way

**FAST NUCES Karachi · Parallel and Distributed Computing · Project #20**

| Member | Roll No. |
|---|---|
| Muhammad Hassan Khan | 23K-0565 |
| Muhammad Uzair Uddin | 23K-0665 |
| Murtaza | 23K-0589 |

Sequential and OpenMP task-parallel **2-way** and **4-way** merge sort, benchmarked on
**large arrays (up to 200 million integers)** with 1–8 threads, plus a **calibrated
analytical performance model** that predicts the run times without being fitted to them.

The proposal was approved with these changes, all implemented:

| Feedback | Where |
|---|---|
| 2-way merge sort | [src/mergesort2.c](src/mergesort2.c) |
| 4-way merge sort | [src/mergesort4.c](src/mergesort4.c) (tournament-tree 4-way merge) |
| Large array sizes | `bench` runs 1M … 200M; any size via `-n` (64-bit indices) |
| Accurate modelling | [scripts/model.py](scripts/model.py) + report Section 6 |

📄 **Report:** [report/PDC_Project_Report.pdf](report/PDC_Project_Report.pdf)

---

## Build

Requires GCC with OpenMP (Linux, macOS with gcc, or Windows MSYS2/MinGW-w64) and
Python 3 with `matplotlib` for the graphs.

```bash
make                 # Linux / MSYS2
mingw32-make         # Windows MinGW
build.bat            # Windows without make
```

## Run

```bash
bin/pdcsort test                         # 5,152 correctness tests against qsort
bin/pdcsort demo -n 50M                  # side-by-side demo (good for the video)
bin/pdcsort bench -n 1M,10M,100M -t 1,2,4,8 -r 3 -o results/bench.csv
bin/pdcsort cutoff -n 50M -t 8 -o results/cutoff.csv
bin/pdcsort variants -n 50M                 # 2-way task-spawning experiment
bin/pdcsort calibrate -o results/calib.csv   # micro-benchmarks for the model
python scripts/analyze.py                # tables + figures + model validation
python scripts/make_report.py            # regenerates the PDF report
bash scripts/run_all.sh                  # everything above (~25 min)
```

Options: `-n` sizes (`K`/`M`/`G` suffixes), `-t` thread counts, `-c` task cutoff(s),
`-r` repetitions (the median is reported), `-o` CSV output file.

## How it works

* **Ping-pong buffers**: one scratch array; each recursion level merges from one buffer
  into the other, so there is no copy-back pass and no allocation during the sort.
* **Leaves** of ≤ 32 elements use insertion sort.
* **2-way parallel**: the first half becomes an `omp task`, the current thread sorts the
  second half, then `taskwait` and `merge2`.
* **4-way parallel**: three quarters become tasks, the current thread sorts the fourth,
  then `taskwait` and `merge4`, a 2-level tournament tree (2 comparisons per element,
  half as many passes over memory as 2-way).
* **Task cutoff** (default 16,384): smaller ranges run the sequential kernel inside one task.
* **Verification**: every timed run is checked for sortedness and for an
  order-independent fingerprint of the input values, which proves the output is a permutation of the input.

## Performance model

`pdcsort calibrate` measures per-thread merge and leaf-sort throughput r(k, q, s) with
q threads running at once, for each merge size s. It also measures task overhead and memory bandwidth.
`scripts/model.py` then predicts the sort level by level:

```
T(n,p) = ceil(k^d / q_d) * L / r_leaf(k, q_d, L)
       + Σ_levels ceil(M_i / q_i) * s_i / r_merge(k, q_i, s_i)
       + N_tasks * t_task / p          where q_i = min(p, M_i)
```

The model captures cache effects, shared-DRAM contention and Hyper-Threading, as well as
the serial top-level merges that limit the speedup. It is compared with every measurement
in `results/summary.csv` (columns `model_s`, `model_err_pct`) and in Figure 8 of the report.

## Repository layout

```
include/           sort.h, util.h
src/               mergesort2.c, mergesort4.c, util.c, main.c
scripts/           model.py, analyze.py, make_report.py, run_all.sh
results/           raw CSVs (bench, cutoff, calib) + summary.csv/json + logs
figures/           PNG graphs used in the report
report/            PDC_Project_Report.pdf
docs/              VIDEO_SCRIPT.md (2–3 min demo voice-over)
```
