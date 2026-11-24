# Demo Video Script (under 2 minutes, one speaker)

Only the report (`PDC_Project_Report.pdf`) and the code (`mergesort.cpp`) are shown; nothing is run.

---

**0:00 – 0:15 · Show: report title page**

> "Hello! This is our PDC project number 20, Parallel Merge Sort using OpenMP tasks.
> We made a 2-way and a 4-way merge sort, both sequential and parallel."

**0:15 – 0:50 · Show: `mergesort.cpp`, scroll to `parMergeSort2Way`, then `parMergeSort4Way`**

> "All our code is in one file. In 2-way, we split the array in two halves.
> The left half becomes an OpenMP task, and the current thread sorts the right half.
> Then `taskwait` waits, and we merge.
> If the part is small, below the cutoff, we sort it normally without making new tasks.
> 4-way is the same idea, but we split into four parts and merge four lists."

**0:50 – 1:05 · Show: report, Section 4 (Experimental Setup table)**

> "We are not running the program again in this video, because it takes about 25 minutes.
> It sorts 7 array sizes up to 200 million numbers, with 5 thread counts, and repeats every run
> 3 times. So we ran it once, and all the results are in the report."

**1:05 – 1:15 · Show: report, Appendix A.1 (correctness table)**

> "For correctness, we compared our output with `std::sort`. All 1,530 tests passed."

**1:15 – 1:35 · Show: report, Section 5.1 table and Figure 1, then Figure 2 (speedup)**

> "For 200 million numbers, sequential takes about 31 seconds and parallel about 11,
> so about 3 times faster. Speedup grows well until 4 threads, because this laptop
> has 4 real cores. After that it grows only a little."

**1:35 – 1:48 · Show: report, Figure 5 (model accuracy)**

> "The last merge runs on one thread, so it limits the speedup.
> Our simple model uses this idea and predicts the time within about 12 percent."

**1:48 – 2:00 · Show: report, Figure 6 (cutoff), then the conclusion**

> "A very small cutoff makes too many tasks and is slow. A medium cutoff works best.
> Both versions are correct and faster in parallel. Thank you!"
