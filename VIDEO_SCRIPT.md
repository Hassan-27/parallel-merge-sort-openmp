# Demo Video Script (2–3 minutes)

Screen-record with OBS Studio, the Xbox Game Bar (Win + Alt + R) or the Windows 11 Snipping Tool
(video mode), and record the voice-over at the same time. Before recording, plug in the charger,
close Chrome and other heavy apps, and build once:
`g++ -O3 -fopenmp -static mergesort.cpp -o mergesort`.
Use a terminal font of 16pt or larger.

Roughly 130–150 words per minute; each block below fits its time slot.

---

### 0:00 – 0:20 · Introduction  *(show README on GitHub, or the report's title page)*

> "Assalam-o-Alaikum. We are Muhammad Hassan Khan, Muhammad Uzair Uddin and Murtaza, and this is
> our PDC project number 20: Parallel Merge Sort using OpenMP tasks. Following our proposal
> feedback, we implemented both a 2-way and a 4-way merge sort, tested them on large arrays of up to
> 200 million integers, and built a simple model that predicts their run time."

### 0:20 – 0:55 · Code walkthrough  *(open `mergesort.cpp`: `parMergeSort2Way`, then `parMergeSort4Way`)*

> "All our code is in one file. In the 2-way version, the left half becomes an OpenMP task, the
> current thread sorts the right half itself, then `taskwait` waits for the task and we merge.
> When a range is smaller than the cutoff of 16 thousand elements, we call the sequential sort
> instead, so we don't create millions of tiny tasks.
>
> The 4-way version splits into four quarters: three tasks plus the current thread. The four
> sorted parts are merged by picking the smallest of the four front elements at each step."

### 0:55 – 1:15 · Correctness  *(terminal: `.\mergesort test`)*

> "First, correctness. The test mode compares every version with `std::sort` on 17 sizes, five
> input types such as sorted, reversed and duplicates, with 1 to 8 threads and two cutoffs.
> All 1,530 tests pass. In the benchmarks, every run is checked against `std::sort` as well."

### 1:15 – 1:55 · Live demo  *(terminal: `.\mergesort 50000000 8 16384 1`)*

> "Now a demo with 50 million random integers and 8 threads. First the two sequential versions…
> then the parallel ones. You can see the time, speedup and efficiency, and PASS means the output
> matches `std::sort`. The speedup is about 3 times, not 8, and the report explains why."

*(If it is slow, cut the wait in editing.)*

### 1:55 – 2:35 · Results and model  *(open the report PDF: Figure 1, then Figure 5)*

> "Here are the full results up to 200 million elements; the dotted lines are our model.
> Each merge runs on one thread, so the chain of merges from a leaf up to the root, the span,
> is serial. We estimate it from the sequential time and apply Amdahl's law. Using only the
> sequential run, the model predicts the parallel times within about 12 percent on average.
> It explains the limit: the final merge of all n elements runs on one thread, and this laptop
> has only 4 physical cores, so the speedup levels off at about 3 times. Sequentially 2-way is a
> bit faster, but in parallel 4-way usually wins, because it creates parallel work faster."

### 2:35 – 2:55 · Cutoff and conclusion  *(report: Figure 6, then the conclusion)*

> "Finally, the cutoff experiment: a cutoff of 32 elements creates millions of tasks and is up to
> three times slower, while anything from about a hundred to millions of elements works well.
> In conclusion, both versions are correct and scale up to the number of cores. A parallel merge
> would be the next step to remove the serial bottleneck. The code and report are on our GitHub.
> Thank you."
