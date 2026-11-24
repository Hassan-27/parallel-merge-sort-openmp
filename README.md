# Parallel Merge Sort using OpenMP Tasks — 2-Way vs 4-Way

**FAST NUCES Karachi · Parallel and Distributed Computing · Project #20**

| Member | Roll No. |
|---|---|
| Muhammad Hassan Khan | 23K-0565 |
| Muhammad Uzair Uddin | 23K-0665 |
| Murtaza | 23K-0589 |

Sequential and OpenMP task-parallel **2-way** and **4-way** merge sort, tested on arrays of
up to **200 million integers** with 1–8 threads.

📄 **Report:** [PDC_Project_Report.pdf](PDC_Project_Report.pdf) (all results, tables and graphs)

## Files

| File | Contents |
|---|---|
| [mergesort.cpp](mergesort.cpp) | all code: merges, sequential and parallel sorts, tests, benchmarks |
| [report.py](report.py) | turns the benchmark results into the PDF report (graphs, tables, model) |
| [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) | script for the demo video |

## Build and run

Needs `g++` with OpenMP. The report also needs Python 3 with `matplotlib`, plus Edge or Chrome.

```bash
g++ -O3 -fopenmp mergesort.cpp -o mergesort    # on MSYS2, add -static if linking fails

./mergesort test                 # correctness tests against std::sort
./mergesort                      # demo: 50M integers, 8 threads
./mergesort 100000000 4 16384 3  # custom: N, threads, cutoff, trials
./mergesort bench                # all experiments -> results/*.csv (about 25 min)
python report.py                 # builds PDC_Project_Report.pdf from results/
```

## Proposal feedback

| Feedback | Where |
|---|---|
| 2-way merge sort | `seqMergeSort2Way`, `parMergeSort2Way` |
| 4-way merge sort | `seqMergeSort4Way`, `parMergeSort4Way` |
| Large array sizes | `bench` runs 1M … 200M |
| Accurate modelling | report Section 6 (work/span + Amdahl's law) |
