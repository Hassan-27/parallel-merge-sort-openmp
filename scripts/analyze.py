"""
analyze.py - turn raw benchmark CSVs into summary tables, model evaluation and figures.

Inputs : results/bench.csv, results/cutoff.csv, results/calib.csv
Outputs: results/summary.csv, results/model_eval.csv, results/summary.json, figures/*.png
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Calibration, predict, span_bound

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- visual style (validated reference palette, fixed slot order) ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WAY_COLOR = {2: BLUE, 4: ORANGE}
SIZE_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 2, "lines.markersize": 5.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def fmt_n(n: int) -> str:
    if n >= 10**9 and n % 10**9 == 0:
        return f"{n // 10**9}G"
    if n >= 10**6 and n % 10**6 == 0:
        return f"{n // 10**6}M"
    if n >= 10**3 and n % 10**3 == 0:
        return f"{n // 10**3}K"
    return str(n)


def load(path):
    data = defaultdict(list)
    allok = True
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["algo"], int(r["n"]), int(r["threads"]), int(r["cutoff"]))
            data[key].append(float(r["time_s"]))
            allok &= r["ok"] == "1"
    return data, allok


def main():
    cal = Calibration()
    bench, bench_ok = load(RES / "bench.csv")
    med = {k: statistics.median(v) for k, v in bench.items()}
    sizes = sorted({k[1] for k in med})
    # sizes drawn as separate lines in the speedup/efficiency charts (<= 6 so colours never repeat)
    plot_sizes = [n for n in sizes if n in (1_000_000, 10_000_000, 50_000_000, 100_000_000)] + [sizes[-1]]
    plot_sizes = sorted(set(plot_sizes))[:6]
    threads = sorted({k[2] for k in med if k[0].startswith("par")})
    cutoff = next(iter(med))[3]

    def T(algo, n, p=1):
        return med[(algo, n, p, cutoff)]

    # ------------------------------------------------------------ summary table
    rows, model_rows = [], []
    for n in sizes:
        best_seq = min(T("seq2", n), T("seq4", n))
        for way in (2, 4):
            seq = T(f"seq{way}", n)
            pred = predict(cal, way, n, 1, cutoff, sequential=True)
            rows.append(dict(algo=f"seq{way}", n=n, threads=1, median_s=seq, speedup=1.0,
                             speedup_vs_best_seq=best_seq / seq, efficiency=1.0,
                             model_s=pred, model_err_pct=100 * (pred - seq) / seq))
            for p in threads:
                t = T(f"par{way}", n, p)
                pred = predict(cal, way, n, p, cutoff)
                rows.append(dict(algo=f"par{way}", n=n, threads=p, median_s=t, speedup=seq / t,
                                 speedup_vs_best_seq=best_seq / t, efficiency=seq / t / p,
                                 model_s=pred, model_err_pct=100 * (pred - t) / t))
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})

    # model accuracy (exclude tiny runs < 0.1 s where 1 ms timer granularity dominates)
    big = [r for r in rows if r["median_s"] >= 0.1]
    mape = statistics.mean(abs(r["model_err_pct"]) for r in big)
    mape_large = statistics.mean(abs(r["model_err_pct"]) for r in big if r["n"] >= 50_000_000)
    within15 = sum(abs(r["model_err_pct"]) <= 15 for r in big) / len(big)

    # ------------------------------------------------------------ cutoff sweep
    cut_rows = []
    if (RES / "cutoff.csv").exists():
        cdata, _ = load(RES / "cutoff.csv")
        for (algo, n, p, c), v in sorted(cdata.items(), key=lambda kv: (kv[0][0], kv[0][3])):
            way = int(algo[-1])
            cut_rows.append(dict(algo=algo, n=n, threads=p, cutoff=c, median_s=statistics.median(v),
                                 model_s=predict(cal, way, n, p, c)))

    # ------------------------------------------------------------ figures
    nmax = sizes[-1]

    # 1. speedup vs threads (two panels: 2-way, 4-way), one line per n
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, way in zip(axes, (2, 4)):
        ax.plot([1, max(threads)], [1, max(threads)], color=INK2, lw=1, ls="--", label="ideal (linear)")
        for i, n in enumerate(plot_sizes):
            sp = [T(f"seq{way}", n) / T(f"par{way}", n, p) for p in threads]
            ax.plot(threads, sp, marker="o", color=SIZE_COLORS[i % 6], label=f"n = {fmt_n(n)}")
        ax.axvline(4, color=GRID, lw=6, zorder=0)
        ax.text(4.1, 0.4, "4 physical cores", color=INK2, fontsize=7.5)
        ax.set_title(f"{way}-way merge sort: speedup")
        ax.set_xlabel("threads")
        ax.set_xticks(threads)
        ax.set_ylim(0, max(threads) + 0.5)
    axes[0].set_ylabel("speedup  (T_seq / T_par)")
    axes[1].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "speedup.png")
    plt.close(fig)

    # 2. efficiency
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, way in zip(axes, (2, 4)):
        for i, n in enumerate(plot_sizes):
            ef = [100 * T(f"seq{way}", n) / T(f"par{way}", n, p) / p for p in threads]
            ax.plot(threads, ef, marker="o", color=SIZE_COLORS[i % 6], label=f"n = {fmt_n(n)}")
        ax.set_title(f"{way}-way merge sort: parallel efficiency")
        ax.set_xlabel("threads")
        ax.set_xticks(threads)
        ax.set_ylim(0, 120)
    axes[0].set_ylabel("efficiency (%)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "efficiency.png")
    plt.close(fig)

    # 3. execution time vs threads for the largest n, 2-way vs 4-way, with model
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for way in (2, 4):
        meas = [T(f"par{way}", nmax, p) for p in threads]
        mod = [predict(cal, way, nmax, p, cutoff) for p in threads]
        ax.plot(threads, meas, marker="o", color=WAY_COLOR[way], label=f"{way}-way measured")
        ax.plot(threads, mod, ls=":", marker="s", mfc="white", color=WAY_COLOR[way], label=f"{way}-way model")
        ax.axhline(T(f"seq{way}", nmax), color=WAY_COLOR[way], lw=1, ls="--", alpha=0.6)
    ax.text(threads[-1], T("seq2", nmax), " seq", va="center", color=INK2, fontsize=7.5)
    ax.set_title(f"Execution time, n = {fmt_n(nmax)} integers")
    ax.set_xlabel("threads")
    ax.set_ylabel("time (s)")
    ax.set_xticks(threads)
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "time_vs_threads.png")
    plt.close(fig)

    # 4. time vs n (log-log) + throughput
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    pmax = max(threads)
    for way in (2, 4):
        c = WAY_COLOR[way]
        axes[0].plot(sizes, [T(f"seq{way}", n) for n in sizes], marker="o", color=c, ls="--", label=f"{way}-way sequential")
        axes[0].plot(sizes, [T(f"par{way}", n, pmax) for n in sizes], marker="o", color=c, label=f"{way}-way, {pmax} threads")
        axes[1].plot(sizes, [n / T(f"par{way}", n, pmax) / 1e6 for n in sizes], marker="o", color=c, label=f"{way}-way, {pmax} threads")
        axes[1].plot(sizes, [n / T(f"seq{way}", n) / 1e6 for n in sizes], marker="o", color=c, ls="--", label=f"{way}-way sequential")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("array size n (integers)")
        ax.set_xticks(sizes)
        ax.set_xticklabels([fmt_n(n) for n in sizes])
        ax.minorticks_off()
    axes[0].set_yscale("log")
    axes[0].set_ylabel("time (s, log scale)")
    axes[0].set_title("Execution time vs array size")
    axes[1].set_ylabel("million elements sorted / s")
    axes[1].set_title("Sorting throughput")
    axes[1].set_ylim(0, None)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "time_vs_n.png")
    plt.close(fig)

    # 5. model vs measured scatter
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    lo, hi = 0.08, max(r["median_s"] for r in rows) * 1.4
    ax.fill_between([lo, hi], [lo * 0.85, hi * 0.85], [lo * 1.15, hi * 1.15], color=GRID, alpha=0.7, lw=0, label="±15% band")
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", label="perfect prediction")
    for way in (2, 4):
        pts = [r for r in big if r["algo"].endswith(str(way))]
        ax.scatter([r["median_s"] for r in pts], [r["model_s"] for r in pts], s=26, color=WAY_COLOR[way],
                   edgecolor="white", linewidth=0.8, label=f"{way}-way", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured time (s)")
    ax.set_ylabel("model-predicted time (s)")
    ax.set_title(f"Model accuracy (MAPE {mape:.1f}%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "model_accuracy.png")
    plt.close(fig)

    # 6. cutoff sweep
    if cut_rows:
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        for way in (2, 4):
            rr = [r for r in cut_rows if r["algo"] == f"par{way}"]
            ax.plot([r["cutoff"] for r in rr], [r["median_s"] for r in rr], marker="o", color=WAY_COLOR[way], label=f"{way}-way measured")
            ax.plot([r["cutoff"] for r in rr], [r["model_s"] for r in rr], marker="s", mfc="white", ls=":", color=WAY_COLOR[way], label=f"{way}-way model")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("task cutoff (elements, log scale)")
        ax.set_ylabel("time (s)")
        ax.set_ylim(0, None)
        r0 = cut_rows[0]
        ax.set_title(f"Effect of task cutoff (n = {fmt_n(r0['n'])}, {r0['threads']} threads)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "cutoff.png")
        plt.close(fig)

    # 6b. task-spawning variants experiment
    var_rows = []
    if (RES / "variants.csv").exists():
        vdata, _ = load(RES / "variants.csv")
        for (algo, n, p, c), v in sorted(vdata.items(), key=lambda kv: (kv[0][0], kv[0][2])):
            var_rows.append(dict(algo=algo, n=n, threads=p, median_s=statistics.median(v)))
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        labels = {"par2": "2-way: spawn 1 half + continue (main)", "par2b": "2-way: spawn both halves",
                  "par4": "4-way: spawn 3 quarters + continue"}
        colors = {"par2": BLUE, "par2b": AQUA, "par4": ORANGE}
        vn = var_rows[0]["n"]
        for algo in ("par2", "par2b", "par4"):
            rr = [r for r in var_rows if r["algo"] == algo]
            ax.plot([r["threads"] for r in rr], [r["median_s"] for r in rr], marker="o", color=colors[algo], label=labels[algo])
        for way in (2, 4):
            ax.plot(threads, [predict(cal, way, vn, p, cutoff) for p in threads], ls=":", color=WAY_COLOR[way], lw=1.5,
                    label=f"{way}-way model (greedy scheduling)")
        ax.set_xlabel("threads")
        ax.set_ylabel("time (s)")
        ax.set_xticks(threads)
        ax.set_ylim(0, None)
        ax.set_title(f"Task-spawning strategy (n = {fmt_n(vn)})")
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(FIG / "variants.png")
        plt.close(fig)

    # 7. calibration: merge throughput per thread vs size
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, way in zip(axes, (2, 4)):
        for i, q in enumerate((1, 2, 4, 8)):
            if (way, q) not in cal.merge:
                continue
            pts = cal.merge[(way, q)]
            ax.plot([s for s, _ in pts], [r / 1e6 for _, r in pts], marker="o", ms=3.5, color=SIZE_COLORS[i], label=f"{q} concurrent thread{'s' if q > 1 else ''}")
        ax.set_xscale("log", base=2)
        ax.set_title(f"merge{way}() throughput per thread")
        ax.set_xlabel("merge output size (elements)")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("M elements / s / thread")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "calibration_merge.png")
    plt.close(fig)

    # 8. model breakdown: where the time goes (largest n, 8 threads)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels, leaf_t, low_t, top_t = [], [], [], []
    for way in (2, 4):
        for p in (1, pmax):
            tot, parts, levels = predict(cal, way, nmax, p, cutoff, breakdown=True)
            top = sum(t for (k, m, s, q, t) in levels if m < p or (p == 1 and k == 0))
            labels.append(f"{way}-way, p={p}")
            leaf_t.append(parts["leaves"])
            top_t.append(top)
            low_t.append(parts["merges"] - top)
    y = range(len(labels))
    ax.barh(y, leaf_t, color=AQUA, height=0.6, label="leaf sub-sorts (parallel)")
    ax.barh(y, low_t, left=leaf_t, color=BLUE, height=0.6, label="merge levels with >= p merges")
    ax.barh(y, top_t, left=[a + b for a, b in zip(leaf_t, low_t)], color=ORANGE, height=0.6, label="top levels (< p merges: idle threads)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("model-predicted time (s)")
    ax.set_title(f"Model time breakdown, n = {fmt_n(nmax)}")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "model_breakdown.png")
    plt.close(fig)

    # ------------------------------------------------------------ json summary for the report
    summary = dict(
        sizes=sizes, threads=threads, cutoff=cutoff, all_verified=bench_ok,
        mape=mape, mape_large=mape_large, within15=within15, n_model_points=len(big),
        rows=rows, cutoff_rows=cut_rows, variant_rows=var_rows,
        mape_by={f"{algo}{'_p>=4' if hi else '_p<4'}": statistics.mean(
            abs(r["model_err_pct"]) for r in big if r["algo"] == algo and (r["threads"] >= 4) == hi)
            for algo in ("seq2", "seq4", "par2", "par4") for hi in (False, True)
            if any(r["algo"] == algo and (r["threads"] >= 4) == hi for r in big)},
        bw={str(k): v for k, v in cal.bw.items()},
        task={str(k): v for k, v in cal.task.items()},
        span={f"{way}": {str(n): dict(T_inf=span_bound(cal, way, n, cutoff),
                                       T_1=predict(cal, way, n, 1, cutoff, sequential=True))
                         for n in sizes} for way in (2, 4)},
    )
    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # console report
    print(f"{'algo':5} {'n':>6} {'p':>2} {'time':>9} {'speedup':>8} {'eff':>6} {'model':>9} {'err':>7}")
    for r in rows:
        print(f"{r['algo']:5} {fmt_n(r['n']):>6} {r['threads']:>2} {r['median_s']:9.4f} {r['speedup']:8.2f} "
              f"{100 * r['efficiency']:5.0f}% {r['model_s']:9.4f} {r['model_err_pct']:+6.1f}%")
    print(f"\nModel MAPE (runs >= 0.1 s): {mape:.1f}%   n>=50M: {mape_large:.1f}%   within ±15%: {100 * within15:.0f}%")
    print(f"All benchmark outputs verified: {bench_ok}")


if __name__ == "__main__":
    main()
