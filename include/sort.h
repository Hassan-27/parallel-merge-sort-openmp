/*
 * sort.h - Public interface for the 2-way and 4-way merge sort implementations.
 *
 * Parallel & Distributed Computing - Project #20
 * Parallel Merge Sort using OpenMP Tasks (2-way and 4-way variants)
 *
 * Every sort function sorts a[0..n) in place (the final result is in `a`),
 * using `tmp` (n ints, caller-allocated) as scratch space.  The recursion uses
 * a "ping-pong" scheme: each level merges from one buffer into the other, so
 * no extra copy-back pass is ever needed.
 */
#ifndef PDC_SORT_H
#define PDC_SORT_H

#include <stddef.h>

/* Sub-arrays at or below this size are finished with insertion sort. */
#define LEAF_SIZE 32

/* Default task cutoff: sub-arrays at or below this size are sorted
 * sequentially inside a single task (no further task creation). */
#define DEFAULT_CUTOFF 16384

/* ---------------- building blocks ---------------- */
void insertion_sort(int *a, size_t n);

/* Merge two sorted runs a[0..na) and b[0..nb) into out[0..na+nb). */
void merge2(const int *a, size_t na, const int *b, size_t nb, int *out);

/* Merge four sorted runs into out[0..n0+n1+n2+n3) with a 2-level
 * tournament tree (2 comparisons per output element). */
void merge4(const int *r0, size_t n0, const int *r1, size_t n1,
            const int *r2, size_t n2, const int *r3, size_t n3, int *out);

/* ---------------- 2-way merge sort ---------------- */
void ms2_seq(int *a, int *tmp, size_t n);
void ms2_par(int *a, int *tmp, size_t n, int threads, size_t cutoff);
/* experiment: spawn both halves as tasks instead of "task + continue" */
void ms2_par_both(int *a, int *tmp, size_t n, int threads, size_t cutoff);

/* ---------------- 4-way merge sort ---------------- */
void ms4_seq(int *a, int *tmp, size_t n);
void ms4_par(int *a, int *tmp, size_t n, int threads, size_t cutoff);

/* Internal sequential kernels, exposed for the calibration micro-benchmarks.
 * Sort a[0..n); if to_tmp != 0 the result ends in t, otherwise in a. */
void ms2_rec(int *a, int *t, size_t n, int to_tmp);
void ms4_rec(int *a, int *t, size_t n, int to_tmp);

#endif
