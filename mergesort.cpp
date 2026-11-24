// ============================================================================
// Parallel Merge Sort using OpenMP Tasks  -  2-Way and 4-Way
// PDC Project #20, FAST NUCES Karachi
//
// Build:  g++ -O3 -fopenmp mergesort.cpp -o mergesort
//
// Usage:
//   mergesort [N] [threads] [cutoff] [trials]   one run of all versions (demo)
//   mergesort test                              correctness tests vs std::sort
//   mergesort bench                             all experiments -> results/*.csv
// ============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <random>
#include <iomanip>
#include <climits>
#include <sys/stat.h>
#include <omp.h>
#ifdef _WIN32
#include <direct.h>
#endif

using namespace std;

const long long DEFAULT_CUTOFF = 16384;

// ============================================================================
// 1. MERGE FUNCTIONS
// ============================================================================

// Merge sorted arr[left..mid] and arr[mid+1..right].
void merge2Way(vector<int>& arr, vector<int>& temp, long long left, long long mid, long long right) {
    long long i = left, j = mid + 1, idx = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[idx++] = arr[i++];
        else                  temp[idx++] = arr[j++];
    }
    while (i <= mid)   temp[idx++] = arr[i++];
    while (j <= right) temp[idx++] = arr[j++];

    for (long long p = left; p <= right; p++) arr[p] = temp[p];
}

// Merge four sorted runs: [left..m1], [m1+1..m2], [m2+1..m3], [m3+1..right].
// Each step picks the smallest of the (up to) four current heads.
void merge4Way(vector<int>& arr, vector<int>& temp, long long left, long long m1, long long m2, long long m3, long long right) {
    long long i = left, j = m1 + 1, k = m2 + 1, l = m3 + 1, idx = left;

    while (i <= m1 || j <= m2 || k <= m3 || l <= right) {
        int minVal = 0;
        int source = 0;

        if (i <= m1) { minVal = arr[i]; source = 1; }
        if (j <= m2 && (source == 0 || arr[j] < minVal)) { minVal = arr[j]; source = 2; }
        if (k <= m3 && (source == 0 || arr[k] < minVal)) { minVal = arr[k]; source = 3; }
        if (l <= right && (source == 0 || arr[l] < minVal)) { minVal = arr[l]; source = 4; }

        if (source == 1)      temp[idx++] = arr[i++];
        else if (source == 2) temp[idx++] = arr[j++];
        else if (source == 3) temp[idx++] = arr[k++];
        else                  temp[idx++] = arr[l++];
    }

    for (long long p = left; p <= right; p++) arr[p] = temp[p];
}

// ============================================================================
// 2. SEQUENTIAL IMPLEMENTATIONS (Baselines)
// ============================================================================

void seqMergeSort2Way(vector<int>& arr, vector<int>& temp, long long left, long long right) {
    if (left >= right) return;
    long long mid = left + (right - left) / 2;
    seqMergeSort2Way(arr, temp, left, mid);
    seqMergeSort2Way(arr, temp, mid + 1, right);
    merge2Way(arr, temp, left, mid, right);
}

void seqMergeSort4Way(vector<int>& arr, vector<int>& temp, long long left, long long right) {
    if (left >= right) return;
    long long quarter = (right - left) / 4;
    long long m1 = left + quarter;
    long long m2 = left + 2 * quarter;
    long long m3 = left + 3 * quarter;

    seqMergeSort4Way(arr, temp, left, m1);
    seqMergeSort4Way(arr, temp, m1 + 1, m2);
    seqMergeSort4Way(arr, temp, m2 + 1, m3);
    seqMergeSort4Way(arr, temp, m3 + 1, right);
    merge4Way(arr, temp, left, m1, m2, m3, right);
}

// ============================================================================
// 3. PARALLEL OPENMP TASK IMPLEMENTATIONS (With Cutoff)
// ============================================================================

void parMergeSort2Way(vector<int>& arr, vector<int>& temp, long long left, long long right, long long cutoff) {
    if (left >= right) return;
    if ((right - left) <= cutoff) {           // small range: no more tasks
        seqMergeSort2Way(arr, temp, left, right);
        return;
    }

    long long mid = left + (right - left) / 2;

    #pragma omp task shared(arr, temp)
    parMergeSort2Way(arr, temp, left, mid, cutoff);

    // Current thread executes the right half itself to save one task
    parMergeSort2Way(arr, temp, mid + 1, right, cutoff);

    #pragma omp taskwait
    merge2Way(arr, temp, left, mid, right);
}

void parMergeSort4Way(vector<int>& arr, vector<int>& temp, long long left, long long right, long long cutoff) {
    if (left >= right) return;
    if ((right - left) <= cutoff) {
        seqMergeSort4Way(arr, temp, left, right);
        return;
    }

    long long quarter = (right - left) / 4;
    long long m1 = left + quarter;
    long long m2 = left + 2 * quarter;
    long long m3 = left + 3 * quarter;

    #pragma omp task shared(arr, temp)
    parMergeSort4Way(arr, temp, left, m1, cutoff);

    #pragma omp task shared(arr, temp)
    parMergeSort4Way(arr, temp, m1 + 1, m2, cutoff);

    #pragma omp task shared(arr, temp)
    parMergeSort4Way(arr, temp, m2 + 1, m3, cutoff);

    parMergeSort4Way(arr, temp, m3 + 1, right, cutoff);

    #pragma omp taskwait
    merge4Way(arr, temp, left, m1, m2, m3, right);
}

// ============================================================================
// 4. HELPERS
// ============================================================================

// Runs one of the four versions on arr[0..N-1].
// algo: "seq2", "seq4", "par2", "par4"
void runSort(const string& algo, vector<int>& arr, vector<int>& temp, int threads, long long cutoff) {
    long long N = arr.size();
    if (algo == "seq2") seqMergeSort2Way(arr, temp, 0, N - 1);
    else if (algo == "seq4") seqMergeSort4Way(arr, temp, 0, N - 1);
    else {
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            {
                if (algo == "par2") parMergeSort2Way(arr, temp, 0, N - 1, cutoff);
                else                parMergeSort4Way(arr, temp, 0, N - 1, cutoff);
            }
        }
    }
}

void makeResultsDir() {
#ifdef _WIN32
    _mkdir("results");
#else
    mkdir("results", 0755);
#endif
}

vector<int> randomArray(long long N, unsigned seed) {
    vector<int> a(N);
    mt19937 rng(seed);
    for (long long i = 0; i < N; i++) a[i] = (int)rng();   // full int range, incl. negatives
    return a;
}

// ============================================================================
// 5. MODE "test": correctness against std::sort
// ============================================================================

int runTests() {
    const vector<long long> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 100, 1000, 4097, 100003, 1000000};
    const vector<string> inputs = {"random", "sorted", "reversed", "few-unique", "all-equal"};
    const vector<int> threadList = {1, 2, 4, 8};
    const vector<long long> cutoffs = {8, DEFAULT_CUTOFF};

    ostringstream out;
    out << "Correctness tests: " << sizes.size() << " sizes x " << inputs.size()
        << " input types x {seq2, seq4, par2, par4 with threads 1,2,4,8 and cutoffs 8, "
        << DEFAULT_CUTOFF << "}\nReference: std::sort, compared element by element.\n\n";

    int totalPass = 0, totalFail = 0;
    for (const string& input : inputs) {
        int pass = 0, fail = 0;
        for (long long N : sizes) {
            vector<int> master = randomArray(N, 42 + N);
            if (input == "sorted")     sort(master.begin(), master.end());
            if (input == "reversed")   sort(master.rbegin(), master.rend());
            if (input == "few-unique") for (int& x : master) x = ((unsigned)x % 10) - 5;
            if (input == "all-equal")  fill(master.begin(), master.end(), 7);
            if (N >= 2 && input == "random") { master[0] = INT_MAX; master[1] = INT_MIN; }

            vector<int> expected = master;
            sort(expected.begin(), expected.end());
            vector<int> arr, temp(N);

            for (string algo : {"seq2", "seq4", "par2", "par4"}) {
                bool parallel = algo[0] == 'p';
                for (int t : (parallel ? threadList : vector<int>{1}))
                    for (long long c : (parallel ? cutoffs : vector<long long>{0})) {
                        arr = master;
                        runSort(algo, arr, temp, t, c);
                        if (arr == expected) pass++;
                        else {
                            fail++;
                            out << "  FAIL: " << algo << " N=" << N << " input=" << input
                                << " threads=" << t << " cutoff=" << c << "\n";
                        }
                    }
            }
        }
        out << "  " << left << setw(12) << input << ": " << setw(4) << pass << " passed, " << fail << " failed\n";
        totalPass += pass;
        totalFail += fail;
    }
    out << "\nTOTAL: " << totalPass << " passed, " << totalFail << " failed  ->  "
        << (totalFail ? "FAILED" : "ALL TESTS PASSED") << "\n";

    cout << out.str();
    makeResultsDir();
    ofstream("results/test.txt") << out.str();
    return totalFail ? 1 : 0;
}

// ============================================================================
// 6. TIMING
// ============================================================================

// Median time of `trials` runs (robust to one disturbed run);
// every run is checked against `expected`.
double timeSort(const string& algo, const vector<int>& master, const vector<int>& expected,
                vector<int>& arr, vector<int>& temp, int threads, long long cutoff, int trials, bool& valid) {
    vector<double> times;
    for (int t = 0; t < trials; t++) {
        arr = master;                                  // reset unsorted input (not timed)
        double start = omp_get_wtime();
        runSort(algo, arr, temp, threads, cutoff);
        times.push_back(omp_get_wtime() - start);
        if (arr != expected) valid = false;
    }
    sort(times.begin(), times.end());
    return times[trials / 2];
}

// ============================================================================
// 7. MODE "bench": all experiments for the report
// ============================================================================

int runBench() {
    const vector<long long> sizes = {1000000, 5000000, 10000000, 25000000, 50000000, 100000000, 200000000};
    const vector<int> threadList = {1, 2, 4, 6, 8};
    const vector<long long> cutoffs = {32, 128, 512, 2048, 8192, 16384, 65536, 262144, 1048576, 4194304};
    const int trials = 3;
    bool valid = true;

    makeResultsDir();
    ofstream csv("results/bench.csv");
    csv << "algo,n,threads,cutoff,time_s\n";

    // --- Experiment 1: time vs array size and threads ---
    for (long long N : sizes) {
        cout << "\n=== N = " << N << " ===\n";
        vector<int> master = randomArray(N, 42);
        vector<int> expected = master;
        sort(expected.begin(), expected.end());
        vector<int> arr(N), temp(N);

        auto record = [&](const string& algo, int threads) {
            double t = timeSort(algo, master, expected, arr, temp, threads, DEFAULT_CUTOFF, trials, valid);
            csv << algo << "," << N << "," << threads << "," << DEFAULT_CUTOFF << "," << t << "\n" << flush;
            cout << "  " << algo << "  threads=" << threads << "  " << fixed << setprecision(4) << t << " s\n";
        };
        record("seq2", 1);
        record("seq4", 1);
        for (int p : threadList) {
            record("par2", p);
            record("par4", p);
        }
    }

    // --- Experiment 2: effect of the task cutoff (N = 50M, 8 threads) ---
    {
        const long long N = 50000000;
        const int p = 8;
        cout << "\n=== Cutoff sweep, N = " << N << ", threads = " << p << " ===\n";
        vector<int> master = randomArray(N, 42);
        vector<int> expected = master;
        sort(expected.begin(), expected.end());
        vector<int> arr(N), temp(N);
        ofstream cut("results/cutoff.csv");
        cut << "algo,n,threads,cutoff,time_s\n";
        for (long long c : cutoffs)
            for (string algo : {"par2", "par4"}) {
                double t = timeSort(algo, master, expected, arr, temp, p, c, trials, valid);
                cut << algo << "," << N << "," << p << "," << c << "," << t << "\n" << flush;
                cout << "  cutoff=" << setw(8) << c << "  " << algo << "  " << fixed << setprecision(4) << t << " s\n";
            }
    }

    cout << "\nAll outputs correct: " << (valid ? "YES" : "NO") << "\n";
    return valid ? 0 : 1;
}

// ============================================================================
// 8. DEFAULT MODE: one size, one thread count (demo)
// ============================================================================

int runDemo(int argc, char* argv[]) {
    long long N      = (argc > 1) ? stoll(argv[1]) : 50000000;      // Default: 50M
    int threads      = (argc > 2) ? stoi(argv[2])  : 8;             // Default: 8 threads
    long long cutoff = (argc > 3) ? stoll(argv[3]) : DEFAULT_CUTOFF;
    int trials       = (argc > 4) ? stoi(argv[4])  : 3;

    cout << "=================================================================\n";
    cout << "N = " << N << " | Threads = " << threads
         << " | Cutoff = " << cutoff << " | Trials = " << trials << "\n";
    cout << "=================================================================\n";

    vector<int> master = randomArray(N, 42);
    vector<int> expected = master;
    sort(expected.begin(), expected.end());
    vector<int> arr(N), temp(N);

    auto show = [&](const string& name, const string& algo, double baseTime) {
        bool valid = true;
        double t = timeSort(algo, master, expected, arr, temp, threads, cutoff, trials, valid);
        double speedup = baseTime > 0 ? baseTime / t : 1.0;
        double eff = baseTime > 0 ? 100.0 * speedup / threads : 100.0;
        cout << left << setw(10) << name << right
             << " | Time: " << fixed << setprecision(4) << setw(8) << t << " s"
             << " | Speedup: " << setprecision(2) << setw(5) << speedup << "x"
             << " | Eff: " << setprecision(1) << setw(5) << eff << "%"
             << " | " << (valid ? "PASS" : "FAIL") << "\n";
        return t;
    };

    double tSeq2 = show("Seq 2-Way", "seq2", 0);
    double tSeq4 = show("Seq 4-Way", "seq4", 0);
    cout << "-----------------------------------------------------------------\n";
    show("Par 2-Way", "par2", tSeq2);
    show("Par 4-Way", "par4", tSeq4);
    return 0;
}

int main(int argc, char* argv[]) {
    string mode = (argc > 1) ? argv[1] : "";
    if (mode == "test")  return runTests();
    if (mode == "bench") return runBench();
    return runDemo(argc, argv);
}
