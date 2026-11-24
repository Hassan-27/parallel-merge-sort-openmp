/*
 * main.c - command line driver for the Parallel Merge Sort project.
 *
 *   pdcsort test                         correctness suite (vs. qsort)
 *   pdcsort demo  [-n 50M] [-t 1,2,4,8]  pretty side-by-side comparison
 *   pdcsort bench [-n 1M,10M,...] [-t 1,2,4,8] [-r 3] [-c 16384] [-o file.csv]
 *   pdcsort cutoff [-n 50M] [-t 8] [-c 1K,4K,...] [-r 3] [-o file.csv]
 *   pdcsort variants [-n 50M] [-t 1,2,4,6,8]  2-way task-spawning experiment
 *   pdcsort calibrate [-o calib.csv] [-t 8]  micro-benchmarks for the model
 *   pdcsort info
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "sort.h"
#include "util.h"

#define MAXLIST 64
#define SEED 20251124ULL

typedef enum { ALG_SEQ2, ALG_SEQ4, ALG_PAR2, ALG_PAR4, ALG_PAR2B } alg_t;
static const char *alg_name[] = { "seq2", "seq4", "par2", "par4", "par2b" };

static void run_alg(alg_t alg, int *a, int *tmp, size_t n, int p, size_t cutoff)
{
    switch (alg) {
    case ALG_SEQ2: ms2_seq(a, tmp, n); break;
    case ALG_SEQ4: ms4_seq(a, tmp, n); break;
    case ALG_PAR2: ms2_par(a, tmp, n, p, cutoff); break;
    case ALG_PAR4: ms4_par(a, tmp, n, p, cutoff); break;
    case ALG_PAR2B: ms2_par_both(a, tmp, n, p, cutoff); break;
    }
}

static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) {
        fprintf(stderr, "error: out of memory allocating %.1f MB\n", bytes / 1e6);
        exit(2);
    }
    return p;
}

/* ------------------------------------------------------------------ args */
typedef struct {
    size_t sizes[MAXLIST]; int nsizes;
    size_t threads[MAXLIST]; int nthreads;
    size_t cutoffs[MAXLIST]; int ncutoffs;
    int reps;
    const char *out;
} opts_t;

static void parse_opts(int argc, char **argv, opts_t *o)
{
    for (int i = 2; i < argc; i++) {
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(argv[i], "-n") && v)      { o->nsizes   = parse_size_list(v, o->sizes,   MAXLIST); i++; }
        else if (!strcmp(argv[i], "-t") && v) { o->nthreads = parse_size_list(v, o->threads, MAXLIST); i++; }
        else if (!strcmp(argv[i], "-c") && v) { o->ncutoffs = parse_size_list(v, o->cutoffs, MAXLIST); i++; }
        else if (!strcmp(argv[i], "-r") && v) { o->reps = atoi(v); i++; }
        else if (!strcmp(argv[i], "-o") && v) { o->out = v; i++; }
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); exit(1); }
    }
}

static FILE *open_out(const char *path, const char *header)
{
    if (!path) return NULL;
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "%s\n", header);
    return f;
}

static double median(double *v, int k)
{
    for (int i = 1; i < k; i++) {               /* tiny insertion sort */
        double x = v[i]; int j = i;
        while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; }
        v[j] = x;
    }
    return (k % 2) ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
}

/* -------------------------------------------------------------- test */
static int cmd_test(void)
{
    const size_t sizes[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                             100, 127, 128, 129, 1000, 1023, 1024, 1025, 4095, 4096, 4097,
                             65537, 100003, 1000000, 3000017 };
    const int nsz = (int)(sizeof sizes / sizeof sizes[0]);
    const int thr[] = { 1, 2, 3, 8 };
    const size_t cut[] = { LEAF_SIZE, 100, DEFAULT_CUTOFF };
    int pass = 0, fail = 0;

    size_t maxn = sizes[nsz - 1];
    int *orig = xmalloc(maxn * sizeof(int));
    int *ref  = xmalloc(maxn * sizeof(int));
    int *a    = xmalloc(maxn * sizeof(int));
    int *tmp  = xmalloc(maxn * sizeof(int));

    printf("Correctness suite: %d sizes x %d distributions x {seq2, seq4, par2, par4 (threads 1,2,3,8; cutoffs 32,100,16384)}\n",
           nsz, (int)DIST_COUNT);
    printf("Reference: C library qsort(); every output is compared element-by-element.\n\n");

    for (int d = 0; d < DIST_COUNT; d++) {
        int dpass = 0, dfail = 0;
        for (int si = 0; si < nsz; si++) {
            size_t n = sizes[si];
            fill_array(orig, n, SEED + n, (dist_t)d);
            memcpy(ref, orig, n * sizeof(int));
            qsort(ref, n, sizeof(int), cmp_int);

            for (int alg = 0; alg < 4; alg++) {
                int np = (alg >= ALG_PAR2) ? 4 : 1;
                int nc = (alg >= ALG_PAR2) ? 3 : 1;
                for (int ti = 0; ti < np; ti++)
                    for (int ci = 0; ci < nc; ci++) {
                        memcpy(a, orig, n * sizeof(int));
                        run_alg((alg_t)alg, a, tmp, n, thr[ti], cut[ci]);
                        if (memcmp(a, ref, n * sizeof(int)) == 0) { pass++; dpass++; }
                        else {
                            fail++; dfail++;
                            printf("  FAIL: %s n=%zu dist=%s threads=%d cutoff=%zu\n",
                                   alg_name[alg], n, dist_name((dist_t)d), thr[ti], cut[ci]);
                        }
                    }
            }
        }
        printf("  %-14s : %4d passed, %d failed\n", dist_name((dist_t)d), dpass, dfail);
    }

    /* explicit extreme values (INT_MIN / INT_MAX) */
    {
        int ext[] = { 2147483647, -2147483647 - 1, 0, 2147483647, -1, 1, -2147483647 - 1, 5 };
        size_t n = sizeof ext / sizeof ext[0];
        memcpy(ref, ext, sizeof ext); qsort(ref, n, sizeof(int), cmp_int);
        for (int alg = 0; alg < 4; alg++) {
            memcpy(a, ext, sizeof ext);
            run_alg((alg_t)alg, a, tmp, n, 4, LEAF_SIZE);
            if (memcmp(a, ref, sizeof ext) == 0) pass++; else { fail++; printf("  FAIL extreme values %s\n", alg_name[alg]); }
        }
        printf("  %-14s : %4d passed\n", "INT_MIN/MAX", 4);
    }

    printf("\nTOTAL: %d passed, %d failed  ->  %s\n", pass, fail, fail ? "FAILED" : "ALL TESTS PASSED");
    free(orig); free(ref); free(a); free(tmp);
    return fail ? 1 : 0;
}

/* -------------------------------------------------------------- timing */
/* Times one sort; regenerates input deterministically (not timed). */
static double time_sort(alg_t alg, int *a, int *tmp, size_t n, int p, size_t cutoff,
                        uint64_t fp_expected, int *ok)
{
    fill_array(a, n, SEED, DIST_UNIFORM);
    double t0 = now_sec();
    run_alg(alg, a, tmp, n, p, cutoff);
    double t1 = now_sec();
    *ok = is_sorted_arr(a, n) && fingerprint(a, n) == fp_expected;
    return t1 - t0;
}

/* -------------------------------------------------------------- demo */
static int cmd_demo(opts_t *o)
{
    size_t n = o->nsizes ? o->sizes[0] : 50000000ULL;
    char sb[32];
    int maxp = omp_get_num_procs();
    if (!o->nthreads) {
        size_t d[] = { 1, 2, 4, 8 }; o->nthreads = 0;
        for (int i = 0; i < 4; i++) if ((int)d[i] <= maxp) o->threads[o->nthreads++] = d[i];
    }

    printf("==================================================================\n");
    printf("  Parallel Merge Sort with OpenMP Tasks  -  2-way vs 4-way\n");
    printf("==================================================================\n");
    printf("  Array size : %s integers (%.0f MB, + same again as scratch)\n", fmt_size(n, sb, sizeof sb), n * 4 / 1e6);
    printf("  Logical CPUs available : %d     Task cutoff : %d elements\n\n", maxp, DEFAULT_CUTOFF);

    int *a = xmalloc(n * sizeof(int)), *tmp = xmalloc(n * sizeof(int));
    touch_pages(tmp, n);
    printf("  Generating random input ... ");
    fflush(stdout);
    fill_array(a, n, SEED, DIST_UNIFORM);
    uint64_t fp = fingerprint(a, n);
    printf("done  (fingerprint %016llx)\n\n", (unsigned long long)fp);

    int ok;
    printf("  [1] Sequential baselines\n");
    double t2 = time_sort(ALG_SEQ2, a, tmp, n, 1, 0, fp, &ok);
    printf("      2-way merge sort : %8.3f s   %s\n", t2, ok ? "verified sorted" : "*** WRONG ***");
    double t4 = time_sort(ALG_SEQ4, a, tmp, n, 1, 0, fp, &ok);
    printf("      4-way merge sort : %8.3f s   %s\n\n", t4, ok ? "verified sorted" : "*** WRONG ***");

    printf("  [2] OpenMP task-parallel versions\n");
    printf("      threads |   2-way time  speedup  eff. |   4-way time  speedup  eff.\n");
    printf("      --------+------------------------------+------------------------------\n");
    for (int i = 0; i < o->nthreads; i++) {
        int p = (int)o->threads[i];
        int ok2, ok4;
        double p2 = time_sort(ALG_PAR2, a, tmp, n, p, DEFAULT_CUTOFF, fp, &ok2);
        double p4 = time_sort(ALG_PAR4, a, tmp, n, p, DEFAULT_CUTOFF, fp, &ok4);
        printf("      %7d | %10.3f s  %6.2fx  %3.0f%% | %10.3f s  %6.2fx  %3.0f%%  %s\n",
               p, p2, t2 / p2, 100 * t2 / p2 / p, p4, t4 / p4, 100 * t4 / p4 / p,
               (ok2 && ok4) ? "ok" : "*** WRONG ***");
        fflush(stdout);
    }
    printf("\n  Speedup = T_sequential / T_parallel,  Efficiency = Speedup / threads\n");
    free(a); free(tmp);
    return 0;
}

/* -------------------------------------------------------------- bench */
static int cmd_bench(opts_t *o)
{
    if (!o->nsizes) { size_t d[] = { 1000000, 10000000, 50000000, 100000000 }; memcpy(o->sizes, d, sizeof d); o->nsizes = 4; }
    if (!o->nthreads) { size_t d[] = { 1, 2, 4, 6, 8 }; memcpy(o->threads, d, sizeof d); o->nthreads = 5; }
    if (o->reps <= 0) o->reps = 3;
    size_t cutoff = o->ncutoffs ? o->cutoffs[0] : DEFAULT_CUTOFF;

    FILE *f = open_out(o->out, "algo,n,threads,cutoff,rep,time_s,ok");
    int allok = 1;
    {   /* warm-up: spin up all cores / OpenMP thread pool, not recorded */
        size_t wn = 10000000;
        int *a = xmalloc(wn * sizeof(int)), *tmp = xmalloc(wn * sizeof(int));
        fill_array(a, wn, SEED, DIST_UNIFORM);
        ms2_par(a, tmp, wn, omp_get_num_procs(), cutoff);
        free(a); free(tmp);
    }
    for (int si = 0; si < o->nsizes; si++) {
        size_t n = o->sizes[si];
        char sb[32];
        /* allocate per size so peak memory is only 2 arrays of the current n */
        int *a = xmalloc(n * sizeof(int)), *tmp = xmalloc(n * sizeof(int));
        touch_pages(a, n); touch_pages(tmp, n);
        fill_array(a, n, SEED, DIST_UNIFORM);
        uint64_t fp = fingerprint(a, n);
        printf("\n=== n = %s ===\n", fmt_size(n, sb, sizeof sb));

        /* configuration list: seq2, seq4, then par2/par4 per thread count.
         * Repetitions are interleaved (rep loop outside the config loop) so
         * slow drifts in CPU clock / background load affect all configs equally. */
        int ncfg = 2 + 2 * o->nthreads;
        if (ncfg > MAXLIST) ncfg = MAXLIST;
        alg_t calg[MAXLIST]; int cp[MAXLIST];
        for (int ci = 0; ci < ncfg; ci++) {
            if (ci == 0) { calg[ci] = ALG_SEQ2; cp[ci] = 1; }
            else if (ci == 1) { calg[ci] = ALG_SEQ4; cp[ci] = 1; }
            else { cp[ci] = (int)o->threads[(ci - 2) / 2]; calg[ci] = ((ci - 2) % 2) ? ALG_PAR4 : ALG_PAR2; }
        }
        int reps = o->reps > 64 ? 64 : o->reps;
        static double times[MAXLIST][64];
        for (int r = 0; r < reps; r++) {
            for (int ci = 0; ci < ncfg; ci++) {
                int ok;
                times[ci][r] = time_sort(calg[ci], a, tmp, n, cp[ci], cutoff, fp, &ok);
                allok &= ok;
                if (f) { fprintf(f, "%s,%zu,%d,%zu,%d,%.6f,%d\n", alg_name[calg[ci]], n, cp[ci], cutoff, r, times[ci][r], ok); fflush(f); }
            }
            printf("  (repetition %d/%d done)\n", r + 1, reps);
            fflush(stdout);
        }
        for (int ci = 0; ci < ncfg; ci++)
            printf("  %-5s threads=%-2d  median %9.4f s\n", alg_name[calg[ci]], cp[ci], median(times[ci], reps));
        fflush(stdout);
        free(a); free(tmp);
    }
    if (f) fclose(f);
    printf("\nAll outputs verified: %s\n", allok ? "YES" : "NO");
    return allok ? 0 : 1;
}

/* -------------------------------------------------------------- cutoff */
static int cmd_cutoff(opts_t *o)
{
    size_t n = o->nsizes ? o->sizes[0] : 50000000ULL;
    int p = o->nthreads ? (int)o->threads[0] : omp_get_num_procs();
    if (!o->ncutoffs) {
        size_t d[] = { 32, 128, 512, 2048, 8192, 16384, 65536, 262144, 1048576, 4194304 };
        memcpy(o->cutoffs, d, sizeof d); o->ncutoffs = 10;
    }
    if (o->reps <= 0) o->reps = 3;
    FILE *f = open_out(o->out, "algo,n,threads,cutoff,rep,time_s,ok");
    int *a = xmalloc(n * sizeof(int)), *tmp = xmalloc(n * sizeof(int));
    touch_pages(a, n); touch_pages(tmp, n);
    fill_array(a, n, SEED, DIST_UNIFORM);
    uint64_t fp = fingerprint(a, n);
    char sb[32];
    printf("Cutoff sweep: n=%s threads=%d\n", fmt_size(n, sb, sizeof sb), p);
    int allok = 1;
    for (int ci = 0; ci < o->ncutoffs; ci++) {
        for (int alg = ALG_PAR2; alg <= ALG_PAR4; alg++) {
            double times[64];
            int reps = o->reps > 64 ? 64 : o->reps;
            for (int r = 0; r < reps; r++) {
                int ok;
                times[r] = time_sort((alg_t)alg, a, tmp, n, p, o->cutoffs[ci], fp, &ok);
                allok &= ok;
                if (f) { fprintf(f, "%s,%zu,%d,%zu,%d,%.6f,%d\n", alg_name[alg], n, p, o->cutoffs[ci], r, times[r], ok); fflush(f); }
            }
            printf("  cutoff=%-8zu %s  median %8.4f s\n", o->cutoffs[ci], alg_name[alg], median(times, reps));
            fflush(stdout);
        }
    }
    if (f) fclose(f);
    free(a); free(tmp);
    return allok ? 0 : 1;
}

/* -------------------------------------------------------------- variants */
/* Scheduling experiment: 2-way "spawn one half + continue" (par2, the main
 * implementation) vs 2-way "spawn both halves" (par2b) vs 4-way (par4). */
static int cmd_variants(opts_t *o)
{
    size_t n = o->nsizes ? o->sizes[0] : 50000000ULL;
    if (!o->nthreads) { size_t d[] = { 1, 2, 4, 6, 8 }; memcpy(o->threads, d, sizeof d); o->nthreads = 5; }
    if (o->reps <= 0) o->reps = 3;
    int reps = o->reps > 64 ? 64 : o->reps;
    FILE *f = open_out(o->out, "algo,n,threads,cutoff,rep,time_s,ok");
    int *a = xmalloc(n * sizeof(int)), *tmp = xmalloc(n * sizeof(int));
    touch_pages(a, n); touch_pages(tmp, n);
    fill_array(a, n, SEED, DIST_UNIFORM);
    uint64_t fp = fingerprint(a, n);
    const alg_t algs[3] = { ALG_PAR2, ALG_PAR2B, ALG_PAR4 };
    static double times[MAXLIST][3][64];
    int allok = 1;
    char sb[32];
    printf("Task-scheduling variants, n=%s\n", fmt_size(n, sb, sizeof sb));
    for (int r = 0; r < reps; r++)
        for (int ti = 0; ti < o->nthreads; ti++)
            for (int k = 0; k < 3; k++) {
                int ok, p = (int)o->threads[ti];
                times[ti][k][r] = time_sort(algs[k], a, tmp, n, p, DEFAULT_CUTOFF, fp, &ok);
                allok &= ok;
                if (f) { fprintf(f, "%s,%zu,%d,%d,%d,%.6f,%d\n", alg_name[algs[k]], n, p, DEFAULT_CUTOFF, r, times[ti][k][r], ok); fflush(f); }
            }
    for (int ti = 0; ti < o->nthreads; ti++)
        printf("  threads=%-2zu  par2 %8.4f s   par2b %8.4f s   par4 %8.4f s\n", o->threads[ti],
               median(times[ti][0], reps), median(times[ti][1], reps), median(times[ti][2], reps));
    if (f) fclose(f);
    free(a); free(tmp);
    return allok ? 0 : 1;
}

/* -------------------------------------------------------------- calibrate */
/*
 * Micro-benchmarks that feed the analytical performance model (scripts/model.py).
 * Nothing here runs the full sort: the model must *predict* the full sort.
 *
 *  merge : per-thread merge throughput (elements/s) when q threads merge
 *          independent runs of total size s concurrently (2-way or 4-way).
 *          Captures cache effects (small s), DRAM bandwidth saturation and
 *          Hyper-Threading contention (large q).
 *  leaf  : per-thread throughput of the sequential kernel on chunks of size s
 *          with q threads running concurrently.
 *  task  : cost of creating + executing one empty OpenMP task.
 *  bw    : aggregate memory copy bandwidth (GB/s) with q threads.
 */
static double calib_merge(int way, int q, size_t s, size_t budget)
{
    double t0 = 0, t1 = 0;
    size_t reps = budget / s; if (reps < 2) reps = 2;
    #pragma omp parallel num_threads(q)
    {
        int *in = xmalloc(s * sizeof(int)), *out = xmalloc(s * sizeof(int));
        fill_array(in, s, SEED + 977 * omp_get_thread_num(), DIST_UNIFORM);
        size_t r = s / (size_t)way;
        for (int k = 0; k < way; k++) {
            size_t len = (k == way - 1) ? s - r * (way - 1) : r;
            qsort(in + k * r, len, sizeof(int), cmp_int);
        }
        memset(out, 0, s * sizeof(int));
        #pragma omp barrier
        #pragma omp master
        t0 = now_sec();
        for (size_t it = 0; it < reps; it++) {
            if (way == 2) merge2(in, r, in + r, s - r, out);
            else merge4(in, r, in + r, r, in + 2 * r, r, in + 3 * r, s - 3 * r, out);
        }
        #pragma omp barrier
        #pragma omp master
        t1 = now_sec();
        free(in); free(out);
    }
    return (double)s * reps / (t1 - t0);
}

static double calib_leaf(int way, int q, size_t s, size_t budget)
{
    double t0 = 0, t1 = 0;
    size_t reps = budget / s; if (reps < 2) reps = 2;
    #pragma omp parallel num_threads(q)
    {
        /* use a pool of distinct chunks so data comes from memory like in the real sort */
        size_t pool = 64;
        int *src = xmalloc(pool * s * sizeof(int));
        int *w = xmalloc(s * sizeof(int)), *t = xmalloc(s * sizeof(int));
        fill_array(src, pool * s, SEED + 31 * omp_get_thread_num(), DIST_UNIFORM);
        memset(w, 0, s * sizeof(int)); memset(t, 0, s * sizeof(int));
        #pragma omp barrier
        #pragma omp master
        t0 = now_sec();
        for (size_t it = 0; it < reps; it++) {
            memcpy(w, src + (it % pool) * s, s * sizeof(int));
            if (way == 2) ms2_rec(w, t, s, (int)(it & 1));
            else          ms4_rec(w, t, s, (int)(it & 1));
        }
        #pragma omp barrier
        #pragma omp master
        t1 = now_sec();
        free(src); free(w); free(t);
    }
    return (double)s * reps / (t1 - t0);
}

static double calib_task(int q)
{
    const int N = 400000;
    volatile int sink = 0;
    double t0 = now_sec();
    #pragma omp parallel num_threads(q)
    {
        #pragma omp single
        {
            for (int i = 0; i < N; i++) {
                #pragma omp task
                { sink += 0; }
            }
            #pragma omp taskwait
        }
    }
    return (now_sec() - t0) / N;
}

static double calib_bw(int q)
{
    const size_t s = 8u << 20;   /* 8M ints = 32 MB per thread, far above L3 */
    double t0 = 0, t1 = 0;
    const int reps = 6;
    #pragma omp parallel num_threads(q)
    {
        int *x = xmalloc(s * sizeof(int)), *y = xmalloc(s * sizeof(int));
        memset(x, 1, s * sizeof(int)); memset(y, 0, s * sizeof(int));
        #pragma omp barrier
        #pragma omp master
        t0 = now_sec();
        for (int r = 0; r < reps; r++) memcpy(y, x, s * sizeof(int));
        #pragma omp barrier
        #pragma omp master
        t1 = now_sec();
        free(x); free(y);
    }
    return 2.0 * s * sizeof(int) * reps * q / (t1 - t0) / 1e9;
}

static int cmd_calibrate(opts_t *o)
{
    int maxq = o->nthreads ? (int)o->threads[0] : omp_get_num_procs();
    FILE *f = open_out(o->out ? o->out : "results/calib.csv", "kind,way,q,size,value");
    printf("Calibrating micro-benchmarks for the performance model (q = 1..%d threads)\n", maxq);

    for (int q = 1; q <= maxq; q++) {
        double bw = calib_bw(q), tk = calib_task(q);
        fprintf(f, "bw,0,%d,0,%.4f\n", q, bw);
        fprintf(f, "task,0,%d,0,%.4e\n", q, tk);
        printf("  q=%d  copy bandwidth %.2f GB/s   task overhead %.2f us\n", q, bw, tk * 1e6);
        fflush(stdout);
    }
    for (int way = 2; way <= 4; way += 2) {
        for (int q = 1; q <= maxq; q++) {
            printf("  way=%d q=%d merge:", way, q);
            for (size_t s = 1u << 10; s <= (8u << 20); s <<= 1) {
                double r = calib_merge(way, q, s, 12u << 20);
                fprintf(f, "merge,%d,%d,%zu,%.6e\n", way, q, s, r);
                printf(" %.0f", r / 1e6);
            }
            printf("  leaf:");
            for (size_t s = 1u << 9; s <= (1u << 17); s <<= 1) {
                double r = calib_leaf(way, q, s, 3u << 20);
                fprintf(f, "leaf,%d,%d,%zu,%.6e\n", way, q, s, r);
                printf(" %.1f", r / 1e6);
            }
            printf("  (M elem/s per thread)\n");
            fflush(stdout); fflush(f);
        }
    }
    fclose(f);
    printf("Calibration written.\n");
    return 0;
}

/* -------------------------------------------------------------- info */
static int cmd_info(void)
{
    printf("OpenMP version macro : %d\n", _OPENMP);
    printf("omp_get_num_procs()  : %d\n", omp_get_num_procs());
    printf("omp_get_max_threads(): %d\n", omp_get_max_threads());
    printf("LEAF_SIZE            : %d\n", LEAF_SIZE);
    printf("DEFAULT_CUTOFF       : %d\n", DEFAULT_CUTOFF);
#ifdef __VERSION__
    printf("Compiler             : %s\n", __VERSION__);
#endif
    return 0;
}

static void usage(void)
{
    puts("usage: pdcsort <test|demo|bench|cutoff|variants|calibrate|info> [options]\n"
         "  -n LIST   array sizes, e.g. 1M,10M,100M (K/M/G decimal suffixes)\n"
         "  -t LIST   thread counts, e.g. 1,2,4,8\n"
         "  -c LIST   task cutoff(s), e.g. 16384 or 1K,16K,256K\n"
         "  -r N      repetitions per configuration (median is reported)\n"
         "  -o FILE   CSV output file");
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(); return 1; }
    opts_t o; memset(&o, 0, sizeof o);
    parse_opts(argc, argv, &o);
    if (!strcmp(argv[1], "test"))      return cmd_test();
    if (!strcmp(argv[1], "demo"))      return cmd_demo(&o);
    if (!strcmp(argv[1], "bench"))     return cmd_bench(&o);
    if (!strcmp(argv[1], "cutoff"))    return cmd_cutoff(&o);
    if (!strcmp(argv[1], "calibrate")) return cmd_calibrate(&o);
    if (!strcmp(argv[1], "variants"))  return cmd_variants(&o);
    if (!strcmp(argv[1], "info"))      return cmd_info();
    usage();
    return 1;
}
