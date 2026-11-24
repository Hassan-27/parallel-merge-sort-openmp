# Demo Video Script (under 2 minutes, one speaker)

**Before recording:** plug in the charger, close heavy apps, build once:
`g++ -O3 -fopenmp -static mergesort.cpp -o mergesort`. Use a big terminal font.

---

**0:00 – 0:15 · Show: report title page**

> "Hello! This is our PDC project number 20, Parallel Merge Sort using OpenMP tasks.
> We made a 2-way and a 4-way merge sort, both sequential and parallel."

**0:15 – 0:45 · Show: `mergesort.cpp`, scroll to `parMergeSort2Way`, then `parMergeSort4Way`**

> "All our code is in one file. In 2-way, we split the array in two halves.
> The left half becomes an OpenMP task, and the current thread sorts the right half.
> Then `taskwait` waits, and we merge.
> If the part is small, below the cutoff, we sort it normally without making new tasks.
> 4-way is the same idea, but we split into four parts and merge four lists."

**0:45 – 1:00 · Show: terminal, run `.\mergesort test`**

> "First we check correctness. We compare our output with `std::sort`
> on many sizes and input types. All 1,530 tests pass."

**1:00 – 1:25 · Show: terminal, run `.\mergesort 50000000 8 16384 1`**

> "Now we sort 50 million numbers with 8 threads.
> First the sequential versions, then the parallel ones.
> The parallel version is about 3 times faster, and PASS means the result is correct."

**1:25 – 1:45 · Show: report, Figure 1 and Figure 2 (speedup)**

> "We tested up to 200 million numbers. Speedup grows well until 4 threads,
> because this laptop has 4 real cores. After that it grows only a little.
> Also, the last merge runs on one thread, so it limits the speedup.
> Our simple model predicts this within about 12 percent."

**1:45 – 2:00 · Show: report, Figure 6 (cutoff), then the conclusion**

> "A very small cutoff makes too many tasks and is slow. A medium cutoff works best.
> So, both versions are correct and faster in parallel. Thank you!"
