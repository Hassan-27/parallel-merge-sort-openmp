/*
 * util.h - data generation, verification, parsing and timing helpers.
 */
#ifndef PDC_UTIL_H
#define PDC_UTIL_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DIST_UNIFORM = 0,   /* uniform over the full 32-bit int range   */
    DIST_SORTED,        /* already ascending                        */
    DIST_REVERSED,      /* strictly descending                      */
    DIST_FEW_UNIQUE,    /* only 16 distinct values (many duplicates)*/
    DIST_ALL_EQUAL,     /* every element identical                  */
    DIST_NEARLY_SORTED, /* ascending with ~1% random swaps          */
    DIST_COUNT
} dist_t;

const char *dist_name(dist_t d);

/* Deterministic generator: the same (n, seed, dist) always produces the
 * same array, independent of the number of threads used to generate it. */
void fill_array(int *a, size_t n, uint64_t seed, dist_t d);

/* Order-independent fingerprint of the multiset of values (used to prove the
 * output is a permutation of the input). */
uint64_t fingerprint(const int *a, size_t n);

/* 1 if a[0..n) is non-decreasing. */
int is_sorted_arr(const int *a, size_t n);

/* Pre-fault pages so first-touch page faults are not timed. */
void touch_pages(int *a, size_t n);

/* Parse "1000", "16K", "50M", "1G" (decimal multipliers) or "16Ki" (binary). */
size_t parse_size(const char *s);
/* Parse a comma separated list of sizes; returns count. */
int parse_size_list(const char *s, size_t *out, int max);

/* Human readable size, e.g. 50000000 -> "50M". */
const char *fmt_size(size_t n, char *buf, size_t buflen);

double now_sec(void);

int cmp_int(const void *x, const void *y);

#endif
