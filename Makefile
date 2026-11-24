# Parallel Merge Sort using OpenMP Tasks (2-way & 4-way)
# Build:  make            (Linux/macOS/MSYS2)   or   mingw32-make   (Windows MinGW)
ifeq ($(origin CC),default)
  CC = gcc
endif
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra -fopenmp
INCLUDE  = -Iinclude
SRC      = src/main.c src/mergesort2.c src/mergesort4.c src/util.c
BIN      = bin/pdcsort

ifeq ($(OS),Windows_NT)
  EXE = .exe
else
  EXE =
endif

all: $(BIN)$(EXE)

$(BIN)$(EXE): $(SRC) include/sort.h include/util.h
	@mkdir -p bin
	$(CC) $(CFLAGS) $(INCLUDE) $(SRC) -o $@

test: all
	./$(BIN)$(EXE) test

demo: all
	./$(BIN)$(EXE) demo -n 50M

bench: all
	@mkdir -p results
	./$(BIN)$(EXE) bench -n 1M,5M,10M,25M,50M,100M,200M -t 1,2,4,6,8 -r 3 -o results/bench.csv

cutoff: all
	@mkdir -p results
	./$(BIN)$(EXE) cutoff -n 50M -t 8 -r 3 -o results/cutoff.csv

variants: all
	@mkdir -p results
	./$(BIN)$(EXE) variants -n 50M -t 1,2,4,6,8 -r 3 -o results/variants.csv

calibrate: all
	@mkdir -p results
	./$(BIN)$(EXE) calibrate -o results/calib.csv

analyze:
	python scripts/analyze.py

clean:
	rm -rf bin

.PHONY: all test demo bench cutoff variants calibrate analyze clean
