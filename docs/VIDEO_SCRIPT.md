# Demo Video Script (2–3 minutes)

Screen-record with OBS Studio, the Xbox Game Bar (Win + Alt + R) or the Windows 11 Snipping Tool
(video mode), and record the voice-over at the same time. Before recording,
plug in the charger, close Chrome and other heavy apps, and build once (`mingw32-make`).
Use a terminal font of 16pt or larger.

Roughly 130–150 words per minute; each block below fits its time slot.

---

### 0:00 – 0:20 · Introduction  *(show README on GitHub, or the report's title page)*

> "Assalam-o-Alaikum. We are Muhammad Hassan Khan, Muhammad Uzair Uddin and Murtaza, and this is
> our PDC project number 20: Parallel Merge Sort using OpenMP tasks. Following our proposal
> feedback, we implemented both a 2-way and a 4-way merge sort, tested them on large arrays of up to
> 200 million integers, and built a performance model that predicts their run time."

### 0:20 – 0:55 · Code walkthrough  *(open `src/mergesort2.c`, then `src/mergesort4.c`)*

> "In the 2-way version, `ms2_task` spawns the first half as an OpenMP task, sorts the second
> half itself, waits with `taskwait`, and merges. Below the cutoff of 16 thousand elements it
> calls the plain sequential function, so we don't create millions of tiny tasks.
> We use two buffers in a ping-pong fashion, so no copy-back is needed.
>
> The 4-way version splits into four quarters: three tasks plus the current thread. The four
> sorted runs are merged with a small tournament tree. Only the pair that lost an element is
> re-compared, so it costs two comparisons per element, but over half as many passes through
> memory."

### 0:55 – 1:15 · Correctness  *(terminal: `bin\pdcsort test`)*

> "First, correctness. The test command compares every version against the C library qsort on
> 33 sizes, six input patterns such as sorted, reversed and duplicates, and several thread counts and
> cutoffs. All 5,152 tests pass. During benchmarking, every run is also checked for sortedness
> and with a fingerprint that proves the output is a permutation of the input."

### 1:15 – 1:55 · Live demo  *(terminal: `bin\pdcsort demo -n 50M`)*

> "Now the demo with 50 million random integers. First the two sequential baselines… then the
> parallel versions with 1, 2, 4 and 8 threads. You can see time, speedup and efficiency
> for both variants. Speedup rises quickly up to 4 threads, which is the number of physical cores
> on this laptop. Beyond that, Hyper-Threading adds only a little, so efficiency drops."

*(While the demo runs, point at the columns. If it is slow, cut the wait in editing.)*

### 1:55 – 2:35 · Results and model  *(open the report PDF: Figure 1, then Figures 8 and 9)*

> "Here are the full results up to 200 million elements. The dotted lines are our model's predictions.
> The model uses only micro-benchmarks: merge throughput measured for different sizes and numbers of
> concurrent threads, plus task overhead. It adds up the time level by level through the recursion
> tree. Its predictions are within a few percent of the measurements on average. The model also
> explains the limit: the top levels have fewer merges than threads, so cores sit idle, and the
> final merge of all n elements runs on a single thread. That serial span caps the speedup."

### 2:35 – 2:55 · Cutoff and conclusion  *(report: cutoff figure, then conclusion)*

> "Finally, the cutoff experiment: very small cutoffs create too many tasks, and very large ones leave
> threads idle. In conclusion, both versions are correct and scale with the number of cores, and
> 4-way merging reduces memory passes. A parallel merge would be the next step to remove the serial
> bottleneck. The code, report and results are on our GitHub. Thank you."
