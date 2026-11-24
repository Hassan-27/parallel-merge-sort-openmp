/*
 * mergesort4.c - 4-way merge sort: sequential and OpenMP-task versions.
 *
 * Recursion tree: every node splits its range into 4 quarters.
 *   depth = log4(n / LEAF_SIZE) = half the depth of the 2-way tree,
 *   so the array is streamed through memory only half as many times.
 *
 * The 4-way merge uses a 2-level tournament tree:
 *
 *            final winner
 *             /        \
 *        win(0,1)    win(2,3)
 *         /   \       /   \
 *       r0    r1    r2    r3
 *
 * After an element is emitted from run w, only the pair containing w is
 * replayed (1 comparison) plus the final match (1 comparison), i.e.
 * 2 comparisons per output element (vs. 1 for 2-way, but over half the
 * number of passes, so the comparison count per element of the whole sort
 * is the same: 2 * log4 n = log2 n).
 *
 * Exhausted runs are represented by a 64-bit key larger than any int, so the
 * inner loop needs no special end-of-run cases and works for the full int
 * range (including INT_MAX and INT_MIN).
 *
 * Parallel version: each node above the cutoff spawns 3 quarters as tasks,
 * works on the 4th itself, waits, then performs the 4-way merge.
 */
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "sort.h"

#define KEY_INF INT64_MAX

void merge4(const int *r0, size_t n0, const int *r1, size_t n1,
            const int *r2, size_t n2, const int *r3, size_t n3, int *out)
{
    const int *p[4] = { r0, r1, r2, r3 };
    const int *e[4] = { r0 + n0, r1 + n1, r2 + n2, r3 + n3 };
    int64_t k[4];
    for (int i = 0; i < 4; i++)
        k[i] = (p[i] < e[i]) ? (int64_t)*p[i] : KEY_INF;

    int w01 = (k[1] < k[0]) ? 1 : 0;
    int w23 = (k[3] < k[2]) ? 3 : 2;

    const size_t total = n0 + n1 + n2 + n3;
    for (size_t o = 0; o < total; o++) {
        int w = (k[w23] < k[w01]) ? w23 : w01;   /* final match */
        out[o] = (int)k[w];
        p[w]++;
        k[w] = (p[w] < e[w]) ? (int64_t)*p[w] : KEY_INF;
        if (w < 2) w01 = (k[1] < k[0]) ? 1 : 0;   /* replay only one pair */
        else       w23 = (k[3] < k[2]) ? 3 : 2;
    }
}

/* Sort a[0..n).  Result goes to t if to_tmp, else stays in a. */
void ms4_rec(int *a, int *t, size_t n, int to_tmp)
{
    if (n <= LEAF_SIZE) {
        insertion_sort(a, n);
        if (to_tmp) memcpy(t, a, n * sizeof(int));
        return;
    }
    size_t q  = n / 4;
    size_t o1 = q, o2 = 2 * q, o3 = 3 * q;
    size_t n3 = n - o3;

    ms4_rec(a,      t,      q,  !to_tmp);
    ms4_rec(a + o1, t + o1, q,  !to_tmp);
    ms4_rec(a + o2, t + o2, q,  !to_tmp);
    ms4_rec(a + o3, t + o3, n3, !to_tmp);

    int *src = to_tmp ? a : t;
    int *dst = to_tmp ? t : a;
    merge4(src, q, src + o1, q, src + o2, q, src + o3, n3, dst);
}

void ms4_seq(int *a, int *tmp, size_t n)
{
    ms4_rec(a, tmp, n, 0);
}

static void ms4_task(int *a, int *t, size_t n, int to_tmp, size_t cutoff)
{
    if (n <= cutoff) {                 /* sequential cutoff */
        ms4_rec(a, t, n, to_tmp);
        return;
    }
    size_t q  = n / 4;
    size_t o1 = q, o2 = 2 * q, o3 = 3 * q;
    size_t n3 = n - o3;

    #pragma omp task firstprivate(a, t, q, to_tmp, cutoff)
    ms4_task(a, t, q, !to_tmp, cutoff);

    #pragma omp task firstprivate(a, t, q, o1, to_tmp, cutoff)
    ms4_task(a + o1, t + o1, q, !to_tmp, cutoff);

    #pragma omp task firstprivate(a, t, q, o2, to_tmp, cutoff)
    ms4_task(a + o2, t + o2, q, !to_tmp, cutoff);

    /* the current thread sorts the last quarter itself */
    ms4_task(a + o3, t + o3, n3, !to_tmp, cutoff);

    #pragma omp taskwait

    int *src = to_tmp ? a : t;
    int *dst = to_tmp ? t : a;
    merge4(src, q, src + o1, q, src + o2, q, src + o3, n3, dst);
}

void ms4_par(int *a, int *tmp, size_t n, int threads, size_t cutoff)
{
    if (cutoff < LEAF_SIZE) cutoff = LEAF_SIZE;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single nowait
        ms4_task(a, tmp, n, 0, cutoff);
    }
}
