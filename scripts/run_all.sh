#!/usr/bin/env bash
# Reproduce every result in the report:  bash scripts/run_all.sh
# (~25 minutes on a 4-core laptop; plug the laptop in and close other apps first)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin results figures

gcc -O3 -march=native -std=c11 -Wall -Wextra -fopenmp -Iinclude \
    src/main.c src/mergesort2.c src/mergesort4.c src/util.c -o bin/pdcsort

./bin/pdcsort test      | tee results/test_log.txt
./bin/pdcsort calibrate -o results/calib.csv | tee results/calib_log.txt
./bin/pdcsort bench  -n 1M,5M,10M,25M,50M,100M,200M -t 1,2,4,6,8 -r 3 -o results/bench.csv | tee results/bench_log.txt
./bin/pdcsort cutoff -n 50M -t 8 -r 3 -o results/cutoff.csv | tee results/cutoff_log.txt
./bin/pdcsort variants -n 50M -t 1,2,4,6,8 -r 3 -o results/variants.csv | tee results/variants_log.txt

python scripts/analyze.py | tee results/analysis_log.txt
python scripts/make_report.py
