/*
 * mergesort2.c - Classic 2-way merge sort: sequential and OpenMP-task versions.
 *
 * Recursion tree: every node splits its range into 2 halves.
 *   depth = log2(n / LEAF_SIZE),  every level streams the whole array once.
 *
 * Parallel version: each internal node whose size exceeds `cutoff` spawns
 * the first half as an OpenMP task and processes the second half itself,
 * then waits (taskwait) and merges.  Below the cutoff the sequential kernel
 * is used, so task-creation overhead is bounded by ~2n/cutoff tasks.
 */
#include <string.h>
#include <omp.h>
#include "sort.h"

void insertion_sort(int *a, size_t n)
{
    for (size_t i = 1; i < n; i++) {
        int x = a[i];
        size_t j = i;
        while (j > 0 && a[j - 1] > x) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = x;
    }
}

void merge2(const int *a, size_t na, const int *b, size_t nb, int *out)
{
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        /* take from a on ties -> stable */
        if (b[j] < a[i]) out[k++] = b[j++];
        else             out[k++] = a[i++];
    }
    if (i < na) memcpy(out + k, a + i, (na - i) * sizeof(int));
    if (j < nb) memcpy(out + k, b + j, (nb - j) * sizeof(int));
}

/* Sort a[0..n).  Result goes to t if to_tmp, else stays in a.
 * The other buffer is used as scratch (ping-pong). */
void ms2_rec(int *a, int *t, size_t n, int to_tmp)
{
    if (n <= LEAF_SIZE) {
        insertion_sort(a, n);
        if (to_tmp) memcpy(t, a, n * sizeof(int));
        return;
    }
    size_t h = n / 2;
    /* children leave their output in the *other* buffer */
    ms2_rec(a,     t,     h,     !to_tmp);
    ms2_rec(a + h, t + h, n - h, !to_tmp);
    if (to_tmp) merge2(a, h, a + h, n - h, t);
    else        merge2(t, h, t + h, n - h, a);
}

void ms2_seq(int *a, int *tmp, size_t n)
{
    ms2_rec(a, tmp, n, 0);
}

static void ms2_task(int *a, int *t, size_t n, int to_tmp, size_t cutoff)
{
    if (n <= cutoff) {               /* sequential cutoff */
        ms2_rec(a, t, n, to_tmp);
        return;
    }
    size_t h = n / 2;

    #pragma omp task firstprivate(a, t, h, to_tmp, cutoff)
    ms2_task(a, t, h, !to_tmp, cutoff);

    /* the current thread works on the second half itself */
    ms2_task(a + h, t + h, n - h, !to_tmp, cutoff);

    #pragma omp taskwait

    if (to_tmp) merge2(a, h, a + h, n - h, t);
    else        merge2(t, h, t + h, n - h, a);
}

/* Variant used in the scheduling experiment (pdcsort variants): BOTH halves
 * are spawned as tasks, so the parent only waits and, while waiting in
 * taskwait, can itself execute one of its own children. */
static void ms2_task_both(int *a, int *t, size_t n, int to_tmp, size_t cutoff)
{
    if (n <= cutoff) {
        ms2_rec(a, t, n, to_tmp);
        return;
    }
    size_t h = n / 2;
    #pragma omp task firstprivate(a, t, h, to_tmp, cutoff)
    ms2_task_both(a, t, h, !to_tmp, cutoff);
    #pragma omp task firstprivate(a, t, h, n, to_tmp, cutoff)
    ms2_task_both(a + h, t + h, n - h, !to_tmp, cutoff);
    #pragma omp taskwait
    if (to_tmp) merge2(a, h, a + h, n - h, t);
    else        merge2(t, h, t + h, n - h, a);
}

void ms2_par_both(int *a, int *tmp, size_t n, int threads, size_t cutoff)
{
    if (cutoff < LEAF_SIZE) cutoff = LEAF_SIZE;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        ms2_task_both(a, tmp, n, 0, cutoff);
    }
}

void ms2_par(int *a, int *tmp, size_t n, int threads, size_t cutoff)
{
    if (cutoff < LEAF_SIZE) cutoff = LEAF_SIZE;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        ms2_task(a, tmp, n, 0, cutoff);
    }
}
