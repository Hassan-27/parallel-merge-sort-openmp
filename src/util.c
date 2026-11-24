/*
 * util.c - data generation, verification, parsing and timing helpers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include "util.h"

#define GEN_BLOCK ((size_t)1 << 16)

static inline uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const char *dist_name(dist_t d)
{
    static const char *names[] = { "uniform", "sorted", "reversed",
                                   "few-unique", "all-equal", "nearly-sorted" };
    return (d >= 0 && d < DIST_COUNT) ? names[d] : "?";
}

void fill_array(int *a, size_t n, uint64_t seed, dist_t d)
{
    long long nb = (long long)((n + GEN_BLOCK - 1) / GEN_BLOCK);

    #pragma omp parallel for schedule(static)
    for (long long b = 0; b < nb; b++) {
        size_t lo = (size_t)b * GEN_BLOCK;
        size_t hi = lo + GEN_BLOCK < n ? lo + GEN_BLOCK : n;
        uint64_t s = seed ^ mix64((uint64_t)b + 0x1234567ULL);
        for (size_t i = lo; i < hi; i++) {
            switch (d) {
            case DIST_UNIFORM:       a[i] = (int)(uint32_t)splitmix64(&s); break;
            case DIST_SORTED:        a[i] = (int)(i - n / 2); break;
            case DIST_REVERSED:      a[i] = (int)(n / 2 - i); break;
            case DIST_FEW_UNIQUE:    a[i] = (int)(splitmix64(&s) % 16) - 8; break;
            case DIST_ALL_EQUAL:     a[i] = 42; break;
            case DIST_NEARLY_SORTED: a[i] = (int)(i - n / 2); break;
            default:                 a[i] = 0;
            }
        }
    }
    if (d == DIST_NEARLY_SORTED && n > 1) {
        uint64_t s = seed;
        size_t swaps = n / 100 + 1;
        for (size_t k = 0; k < swaps; k++) {
            size_t i = splitmix64(&s) % n, j = splitmix64(&s) % n;
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
}

uint64_t fingerprint(const int *a, size_t n)
{
    uint64_t h = 0;
    long long nn = (long long)n;
    #pragma omp parallel for reduction(+:h) schedule(static)
    for (long long i = 0; i < nn; i++)
        h += mix64((uint64_t)(uint32_t)a[i] + 0x9E3779B97F4A7C15ULL);
    return h;
}

int is_sorted_arr(const int *a, size_t n)
{
    if (n < 2) return 1;
    int bad = 0;
    long long nn = (long long)n;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (long long i = 1; i < nn; i++)
        if (a[i - 1] > a[i]) bad = 1;
    return !bad;
}

void touch_pages(int *a, size_t n)
{
    long long nn = (long long)n;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < nn; i += 1024) a[i] = 0;
}

size_t parse_size(const char *s)
{
    char *end;
    double v = strtod(s, &end);
    size_t mul = 1;
    if (*end) {
        char c = (char)toupper((unsigned char)*end);
        int bin = end[1] == 'i' || end[1] == 'I';
        if (c == 'K') mul = bin ? 1024ULL : 1000ULL;
        else if (c == 'M') mul = bin ? 1048576ULL : 1000000ULL;
        else if (c == 'G') mul = bin ? 1073741824ULL : 1000000000ULL;
    }
    return (size_t)(v * (double)mul + 0.5);
}

int parse_size_list(const char *s, size_t *out, int max)
{
    char buf[1024];
    strncpy(buf, s, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    int cnt = 0;
    for (char *tok = strtok(buf, ","); tok && cnt < max; tok = strtok(NULL, ","))
        out[cnt++] = parse_size(tok);
    return cnt;
}

const char *fmt_size(size_t n, char *buf, size_t len)
{
    if (n >= 1000000000ULL && n % 1000000000ULL == 0) snprintf(buf, len, "%zuG", n / 1000000000ULL);
    else if (n >= 1000000ULL && n % 1000000ULL == 0)  snprintf(buf, len, "%zuM", n / 1000000ULL);
    else if (n >= 1000ULL && n % 1000ULL == 0)        snprintf(buf, len, "%zuK", n / 1000ULL);
    else snprintf(buf, len, "%zu", n);
    return buf;
}

double now_sec(void) { return omp_get_wtime(); }

int cmp_int(const void *x, const void *y)
{
    int a = *(const int *)x, b = *(const int *)y;
    return (a > b) - (a < b);
}
