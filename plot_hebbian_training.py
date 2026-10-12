#!/usr/bin/env python3
"""Plot training progress from train_hebbian.py's generation_log_<jobid>.csv.

Usage (from the project folder, where logs/ is):

    python3 plot_hebbian_training.py
        -> plots the most recent logs/generation_log_*.csv
    python3 plot_hebbian_training.py --csv logs/generation_log_3215600.csv
        -> plots one specific run
    python3 plot_hebbian_training.py --compare logs/generation_log_3205173.csv logs/generation_log_3215600.csv \
                                     --labels "ABCD only" "weights + ABCD"
        -> overlays several runs on the same axes (survival, best fitness, ...)

Output: a PNG next to where you run it, named after the job id
(training_progress_<jobid>.png, or comparison_<id1>_<id2>.png).
A short text summary is printed as well.

Works with old and new CSVs: panels whose column is missing from a log
(e.g. sat_frac in a run from before it was added) are shown as "not logged".

Needs only matplotlib (pip install --user matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")  # write straight to a file, no display needed over SSH
import matplotlib.pyplot as plt

# Fixed categorical colors (assigned in this order, never cycled).
C1 = "#2a78d6"  # blue
C2 = "#eb6834"  # orange
C3 = "#1baf7a"  # aqua
C4 = "#eda100"  # yellow
C5 = "#e87ba4"  # magenta
C6 = "#008300"  # green
C7 = "#6250d6"  # violet
C8 = "#e34948"  # red
SERIES = [C1, C2, C3, C4, C5, C6, C7, C8]
INK = "#4a4a48"     # axis text
GRID = "#d9d8d4"    # recessive grid

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": "#1f1f1e",
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- loading
def load(csv_path: str) -> dict[str, list[float]]:
    """Read the CSV into {column: [values per generation]}. Non-numeric cells are skipped."""
    cols: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    return cols


def job_id(csv_path: str) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return base.replace("generation_log", "").lstrip("_") or "run"


def latest_csv() -> str | None:
    files = glob.glob("logs/generation_log*.csv") + glob.glob("generation_log*.csv")
    return max(files, key=os.path.getmtime) if files else None


def has(d, *keys):
    return all(k in d and len(d[k]) > 0 for k in keys)


def missing(ax, title):
    ax.set_title(title)
    ax.text(0.5, 0.5, "not logged in this run", ha="center", va="center",
            transform=ax.transAxes, color=INK, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def style(ax, title, ylabel):
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("generation")
    ax.grid(color=GRID, linewidth=0.6)


# ---------------------------------------------------------------- single run
def plot_single(d, run, out_path, episode_steps):
    g = d["generation"]
    fig, axes = plt.subplots(5, 2, figsize=(12, 17.5))
    fig.suptitle(f"Hebbian / CMA-ES training, job {run}", fontsize=12, x=0.02, ha="left")
    a = axes.ravel()

    # 1. fitness
    ax = a[0]
    if has(d, "fitness_mean", "fitness_best"):
        m, s = d["fitness_mean"], d.get("fitness_std", [0] * len(g))
        ax.fill_between(g, [x - y for x, y in zip(m, s)], [x + y for x, y in zip(m, s)],
                        color=C1, alpha=0.15, linewidth=0, label="mean ± 1 std")
        ax.plot(g, m, color=C1, lw=2, label="mean")
        ax.plot(g, d["fitness_best"], color=C2, lw=2, label="best genome")
        style(ax, "Fitness (per drone, max 100)", "fitness")
        ax.legend(loc="upper left")
    else:
        missing(ax, "Fitness")

    # 2. steps survived
    ax = a[1]
    if has(d, "steps_survived_mean"):
        ax.plot(g, d["steps_survived_mean"], color=C1, lw=2, label="mean over genomes")
        if has(d, "steps_survived_best"):
            ax.plot(g, d["steps_survived_best"], color=C2, lw=2, label="best genome")
        ax.axhline(episode_steps, color=INK, lw=1, ls="--")
        ax.text(g[0], episode_steps, " full episode", va="bottom", fontsize=8, color=INK)
        ax.set_ylim(0, episode_steps * 1.08)
        style(ax, "Steps survived until first crash (avg of reps)", "steps (100 = 1 s)")
        ax.legend(loc="lower right")
    else:
        missing(ax, "Steps survived")

    # 3. full episodes
    ax = a[2]
    if has(d, "frac_full_episode"):
        ax.plot(g, [100 * x for x in d["frac_full_episode"]], color=C3, lw=2)
        ax.set_ylim(bottom=0)
        style(ax, "Evaluations that never crashed", "% of all swarms")
    else:
        missing(ax, "Evaluations that never crashed")

    # 4. saturation
    ax = a[3]
    if has(d, "sat_frac"):
        ax.plot(g, [100 * x for x in d["sat_frac"]], color=C8, lw=2)
        ax.set_ylim(bottom=0)
        style(ax, "Commands at the action limits (>95%)", "% of actions")
    else:
        missing(ax, "Commands at the action limits")

    # 5. weight size
    ax = a[4]
    if has(d, "w_abs_end"):
        if has(d, "w_abs_start"):
            ax.plot(g, d["w_abs_start"], color=C7, lw=2, label="before episode (evolved start)")
        ax.plot(g, d["w_abs_end"], color=C4, lw=2, label="end of episode (after plasticity)")
        style(ax, "Mean |weight|, layer 1", "mean |w|")
        ax.legend(loc="upper left")
    else:
        missing(ax, "Mean |weight|")

    # 6. sigma
    ax = a[5]
    if has(d, "cma_sigma"):
        ax.plot(g, d["cma_sigma"], color=C5, lw=2)
        style(ax, "CMA-ES step size (sigma)", "sigma")
    else:
        missing(ax, "CMA-ES sigma")

    # 7. genome variance
    ax = a[6]
    if has(d, "genome_variance_mean"):
        ax.plot(g, d["genome_variance_mean"], color=C6, lw=2)
        style(ax, "Population genome variance (mean over parameters)", "variance")
    else:
        missing(ax, "Genome variance")

    # 8. time per generation
    ax = a[7]
    if has(d, "seconds"):
        ax.plot(g, d["seconds"], color=INK, lw=1.5)
        ax.set_ylim(bottom=0)
        style(ax, "Wall-clock time per generation", "seconds")
    else:
        missing(ax, "Time per generation")

    # 9. command size
    ax = a[8]
    if has(d, "act_acc_frac", "act_rate_frac"):
        ax.plot(g, [100 * x for x in d["act_acc_frac"]], color=C1, lw=2, label="linear acceleration (limit 5 m/s²)")
        ax.plot(g, [100 * x for x in d["act_rate_frac"]], color=C2, lw=2, label="body rate (limit 2 rad/s)")
        ax.set_ylim(bottom=0)
        style(ax, "Mean command size", "% of the limit")
        ax.legend(loc="upper left")
    else:
        missing(ax, "Mean command size")

    # 10. first-crash reasons
    ax = a[9]
    reasons = [k for k in d if k.startswith("crash_")]
    if reasons:
        for i, k in enumerate(reasons):
            vals = [100 * x for x in d[k]]
            if max(vals) > 0:  # only conditions that ever ended a swarm
                ax.plot(g, vals, color=SERIES[i % len(SERIES)], lw=1.8, label=k[len("crash_"):])
        ax.set_ylim(0, 105)
        style(ax, "What ended each swarm's first crash", "% of crashed swarms")
        ax.legend(loc="upper left", ncol=2)
    else:
        missing(ax, "First-crash reasons")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- comparison
COMPARE_PANELS = [
    ("steps_survived_mean", "Steps survived until first crash (mean over genomes)", "steps", 1),
    ("steps_survived_best", "Steps survived until first crash (best genome)", "steps", 1),
    ("fitness_best", "Best fitness (per drone, max 100)", "fitness", 1),
    ("frac_full_episode", "Evaluations that never crashed", "% of all swarms", 100),
]


def plot_compare(runs, labels, out_path, episode_steps):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Run comparison", fontsize=12, x=0.02, ha="left")
    for ax, (key, title, ylabel, factor) in zip(axes.ravel(), COMPARE_PANELS):
        drawn = False
        for i, (d, label) in enumerate(zip(runs, labels)):
            if has(d, key):
                ax.plot(d["generation"], [factor * x for x in d[key]],
                        color=SERIES[i % len(SERIES)], lw=2, label=label)
                drawn = True
        if not drawn:
            missing(ax, title)
            continue
        if key.startswith("steps_survived"):
            ax.axhline(episode_steps, color=INK, lw=1, ls="--")
            ax.set_ylim(0, episode_steps * 1.08)
        style(ax, title, ylabel)
        ax.legend(loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- summary
def summary(d, run):
    def first_last(k, fmt="{:.2f}"):
        if not has(d, k):
            return "not logged"
        return f"{fmt.format(d[k][0])} -> {fmt.format(d[k][-1])}"

    n = len(d.get("generation", []))
    print(f"\n--- summary, job {run} ---")
    print(f"generations           : {n}")
    print(f"fitness mean          : {first_last('fitness_mean')}")
    if has(d, "fitness_best"):
        print(f"fitness best (max)    : {max(d['fitness_best']):.2f}")
    print(f"steps survived mean   : {first_last('steps_survived_mean', '{:.0f}')}")
    if has(d, "steps_survived_best"):
        print(f"steps survived best   : {max(d['steps_survived_best']):.0f} (max over run)")
    if has(d, "frac_full_episode"):
        print(f"never crashed         : {100 * d['frac_full_episode'][-1]:.1f}% (last gen), "
              f"{100 * max(d['frac_full_episode']):.1f}% (max)")
    if has(d, "sat_frac"):
        print(f"saturated actions     : {100 * d['sat_frac'][0]:.1f}% -> {100 * d['sat_frac'][-1]:.1f}%")
    print(f"mean |w| start        : {first_last('w_abs_start', '{:.3f}')}")
    print(f"mean |w| end          : {first_last('w_abs_end', '{:.3f}')}")
    if has(d, "act_acc_frac", "act_rate_frac"):
        print(f"command size (% limit): acc {100 * d['act_acc_frac'][-1]:.1f}%, "
              f"rate {100 * d['act_rate_frac'][-1]:.1f}% (last gen)")
    reasons = {k[len("crash_"):]: d[k][-1] for k in d if k.startswith("crash_") and d[k]}
    if reasons:
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        print("first-crash reasons   : " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in top if v > 0)
              + " (last gen)")
    print(f"sigma                 : {first_last('cma_sigma', '{:.4f}')}")
    if has(d, "seconds"):
        print(f"seconds/gen           : {d['seconds'][0]:.0f} -> {d['seconds'][-1]:.0f} "
              f"(mean {sum(d['seconds']) / n:.0f})")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Plot Hebbian/CMA-ES training logs")
    p.add_argument("--csv", help="one generation_log_<jobid>.csv (default: the newest in logs/)")
    p.add_argument("--compare", nargs="+", help="several CSVs to overlay")
    p.add_argument("--labels", nargs="+", help="legend labels for --compare (default: job ids)")
    p.add_argument("--episode_steps", type=int, default=2000, help="max episode length (for the dashed line)")
    p.add_argument("--out", help="output PNG path (default: named after the job id)")
    args = p.parse_args()

    if args.compare:
        runs = [load(c) for c in args.compare]
        ids = [job_id(c) for c in args.compare]
        labels = args.labels if args.labels and len(args.labels) == len(ids) else ids
        out = args.out or f"comparison_{'_'.join(ids)}.png"
        plot_compare(runs, labels, out, args.episode_steps)
        for d, i in zip(runs, ids):
            summary(d, i)
        print(f"\n[INFO] Saved {out}")
        return

    path = args.csv or latest_csv()
    if path is None:
        p.error("No generation_log*.csv found in ./logs. Pass --csv explicitly.")
    if not args.csv:
        print(f"[INFO] Using newest log: {path}")
    d = load(path)
    run = job_id(path)
    out = args.out or f"training_progress_{run}.png"
    plot_single(d, run, out, args.episode_steps)
    summary(d, run)
    print(f"\n[INFO] Saved {out}")


if __name__ == "__main__":
    main()