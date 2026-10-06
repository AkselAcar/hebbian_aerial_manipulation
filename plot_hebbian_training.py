#!/usr/bin/env python3
"""Plot training progress from train_hebbian.py's generation_log*.csv.

Usage (on the cluster, or locally after copying the CSV over):
    python3 plot_hebbian_training.py
        -> with no --csv, auto-finds the most recently modified
           generation_log*.csv in ./logs (or the current directory) and
           uses that. This is the normal way to run it: after a job
           finishes (or while it's still running), just run this with no
           arguments and it plots your latest run.
    python3 plot_hebbian_training.py --csv logs/generation_log_3194838.csv
        -> plot a specific run by its SLURM job id (see train_hebbian.py's
           per-run CSV naming) instead of the newest one -- e.g. to look
           back at an older attempt.
    python3 plot_hebbian_training.py --all-runs
        -> if you point --csv at an OLD-style log that has multiple runs
           appended into one file, overlay all of them instead of just
           the latest.

Needs only matplotlib (pip install --user matplotlib) -- CSV parsing uses the
standard library, on purpose, so you don't need pandas or any cluster module
you don't already have.

--------------------------------------------------------------------------
Why this exists / what it plots
--------------------------------------------------------------------------
Fuda's feedback on the first (flat) curve was: "I would also plot the
standard deviation of the fitness, and mean variance of your genotype."
This script produces exactly those, alongside the fitness curve itself:

    Panel 1 - fitness per generation: best genome (line) and population
              mean (line), with the +/- 1 std band shaded around the mean.
              This is the "is it learning" plot, now with the spread Fuda
              asked for instead of just a single flat-looking line.
    Panel 2 - fitness standard deviation per generation, on its own axis
              (the shaded band in panel 1 shows this too, but Fuda asked
              for it explicitly as its own plot, so it's broken out here).
    Panel 3 - mean genotype variance per generation. A shrinking value
              here means CMA-ES's search distribution is narrowing
              (converging); if it stays flat/high, the population isn't
              settling on anything.
    Panel 4 (bonus, not explicitly requested but explains 2 and 3
              together) - CMA-ES's sigma (its own internal step size).
              Sigma and genotype variance should decay together as the
              search converges; if they diverge, something's off.

--------------------------------------------------------------------------
Handling multiple runs in one log file
--------------------------------------------------------------------------
train_hebbian.py opens the CSV in *append* mode, so if you stop a job and
resubmit, the new run's "generation 0" gets appended right after the old
run's last row -- the generation column just resets partway through the
file. This script detects those resets (a generation number that drops
back down) and splits the log into separate runs. By default it plots
only the LAST (most recent) run, since that's almost always what you
want; pass --all-runs to overlay every run instead (useful for comparing
attempts against each other).
"""

from __future__ import annotations  # lets str | None / list[dict] hints work on Python < 3.10

import argparse
import csv
import datetime
import glob
import os

import matplotlib

matplotlib.use("Agg")  # write straight to a file; no display/X11 needed over SSH
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (fixed assignment, not cycled).
COLOR_BEST = "#D55E00"      # vermillion
COLOR_MEAN = "#0072B2"      # blue
COLOR_STD_BAND = "#0072B2"  # same hue as mean, used at low alpha
COLOR_STD_LINE = "#D55E00"  # vermillion, panel 2
COLOR_VARIANCE = "#009E73"  # bluish green
COLOR_SIGMA = "#CC79A7"     # reddish purple

FIELDS = ["generation", "fitness_mean", "fitness_std", "fitness_best",
          "genome_variance_mean", "cma_sigma", "seconds"]


def load_rows(csv_path: str) -> list[dict]:
    """Read generation_log.csv into a list of dicts with float values."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(r[k]) for k in FIELDS})
    return rows


def split_into_runs(rows: list[dict]) -> list[list[dict]]:
    """Split an append-mode log into separate runs.

    A new run starts wherever `generation` does not strictly increase
    from the previous row (i.e. it dropped back to 0, or to some other
    value <= the row before it).
    """
    runs = []
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["generation"] <= rows[i - 1]["generation"]:
            runs.append(rows[start:i])
            start = i
    runs.append(rows[start:])
    return runs


def col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows]


def plot_run(run: list[dict], out_path: str, title_suffix: str = ""):
    gen = col(run, "generation")
    fitness_mean = col(run, "fitness_mean")
    fitness_std = col(run, "fitness_std")
    fitness_best = col(run, "fitness_best")
    genome_var = col(run, "genome_variance_mean")
    sigma = col(run, "cma_sigma")

    lower = [m - s for m, s in zip(fitness_mean, fitness_std)]
    upper = [m + s for m, s in zip(fitness_mean, fitness_std)]

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    # --- Panel 1: fitness (best + mean +/- std) ---
    ax = axes[0]
    ax.plot(gen, fitness_best, color=COLOR_BEST, linewidth=1.8, label="best fitness")
    ax.plot(gen, fitness_mean, color=COLOR_MEAN, linewidth=1.8, label="mean fitness")
    ax.fill_between(gen, lower, upper, color=COLOR_STD_BAND, alpha=0.2, label="mean +/- 1 std")
    ax.set_ylabel("fitness")
    ax.set_xlabel("generation")
    ax.set_title(f"Fitness per generation{title_suffix}")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)

    # --- Panel 2: fitness standard deviation (explicitly requested by Fuda) ---
    ax = axes[1]
    ax.plot(gen, fitness_std, color=COLOR_STD_LINE, linewidth=1.8)
    ax.set_ylabel("fitness std dev")
    ax.set_xlabel("generation")
    ax.set_title("Standard deviation of fitness across the population")
    ax.grid(alpha=0.3)

    # --- Panel 3: mean genotype variance (explicitly requested by Fuda) ---
    ax = axes[2]
    ax.plot(gen, genome_var, color=COLOR_VARIANCE, linewidth=1.8)
    ax.set_ylabel("mean genotype variance")
    ax.set_xlabel("generation")
    ax.set_title("Population genotype variance per generation")
    ax.grid(alpha=0.3)

    # --- Panel 4 (bonus): CMA-ES sigma ---
    ax = axes[3]
    ax.plot(gen, sigma, color=COLOR_SIGMA, linewidth=1.8)
    ax.set_ylabel("CMA-ES sigma")
    ax.set_xlabel("generation")
    ax.set_title("CMA-ES step size (sigma) -- for context alongside genotype variance")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mean(xs):
    return sum(xs) / len(xs)


def find_latest_csv() -> str | None:
    """Auto-detect the most recently written generation_log*.csv.

    Looks in both ./logs (if you run this from the project root, next to
    the logs/ folder) and the current directory (if you run this from
    inside logs/ itself) -- covers either place you might put the script.
    Matches both the new per-run naming (generation_log_<jobid>.csv) and
    the old single-file naming (generation_log.csv).
    """
    candidates = []
    for pattern in ("logs/generation_log*.csv", "generation_log*.csv"):
        candidates.extend(glob.glob(pattern))
    candidates = list(set(candidates))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def timestamped(path: str) -> str:
    """Insert a YYYYMMDD_HHMMSS timestamp before the extension, so repeated
    runs of this script never overwrite a previous plot. Used as a fallback
    for old-style generation_log.csv files that have no job id in their name."""
    base, ext = (path.rsplit(".", 1) + ["png"])[:2]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"


def derive_out_name(csv_path: str, out_template: str) -> str:
    """Name the plot after the CSV it came from, not after when you happened
    to run this script -- e.g. logs/generation_log_3212345.csv with
    --out training_progress.png becomes training_progress_3212345.png. This
    way the filename itself tells you which job the plot belongs to."""
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]  # "generation_log_3212345"
    suffix = csv_base
    if suffix.startswith("generation_log"):
        suffix = suffix[len("generation_log"):].lstrip("_")  # "3212345", or "" for old-style file
    base, ext = (out_template.rsplit(".", 1) + ["png"])[:2]
    if suffix:
        return f"{base}_{suffix}.{ext}"
    # Old-style generation_log.csv has no job id to borrow -- fall back to a timestamp
    # so we still don't silently overwrite a previous plot.
    return timestamped(out_template)


def main():
    parser = argparse.ArgumentParser(description="Plot Hebbian/CMA-ES training progress")
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to a specific generation_log*.csv. If omitted, auto-detects the "
                              "most recently modified generation_log*.csv in ./logs or the current "
                              "directory (i.e. your latest run) -- this is the normal way to use it.")
    parser.add_argument("--out", type=str, default="training_progress.png",
                         help="Output image path (default: training_progress.png). The CSV's job id is "
                              "always inserted before the extension (e.g. training_progress_3212345.png) "
                              "so the filename tells you which run it's from and previous plots are never "
                              "overwritten; with --all-runs, the run number is added too.")
    parser.add_argument("--all-runs", action="store_true",
                         help="Plot every run found in the log separately, not just the latest one.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Disable the job-id/timestamp naming and write exactly to --out, "
                              "overwriting it if it exists.")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    if csv_path is None:
        parser.error("No generation_log*.csv found in ./logs or the current directory. "
                      "Pass --csv path/to/generation_log_<jobid>.csv explicitly.")
    if args.csv is None:
        print(f"[INFO] No --csv given -- auto-detected latest: {csv_path}")

    rows = load_rows(csv_path)
    runs = split_into_runs(rows)

    if len(runs) > 1:
        print(f"[INFO] Detected {len(runs)} separate runs in {csv_path} "
              f"(generation numbers reset {len(runs) - 1} time(s), most likely from resubmitted jobs).")
        for i, run in enumerate(runs):
            print(f"        run {i}: {len(run)} generations, "
                  f"avg {mean(col(run, 'seconds')):.1f}s/gen, "
                  f"fitness_mean {run[0]['fitness_mean']:.2f} -> {run[-1]['fitness_mean']:.2f}")

    out = args.out if args.overwrite else derive_out_name(csv_path, args.out)

    if args.all_runs:
        base, ext = out.rsplit(".", 1) if "." in out else (out, "png")
        for i, run in enumerate(runs):
            out_path = f"{base}_run{i}.{ext}"
            plot_run(run, out_path, title_suffix=f" (run {i})")
            print(f"[INFO] Saved {out_path}")
    else:
        latest = runs[-1]
        if len(runs) > 1:
            print(f"[INFO] Plotting only the latest run (run {len(runs) - 1}). "
                  f"Pass --all-runs to plot every run.")
        plot_run(latest, out)
        print(f"[INFO] Saved {out}")

        # Quick numeric summary, handy for a status update / email.
        print("\n--- summary (latest run) ---")
        print(f"generations: {len(latest)}")
        print(f"fitness_mean: {latest[0]['fitness_mean']:.2f} -> {latest[-1]['fitness_mean']:.2f}")
        print(f"fitness_best (max over run): {max(col(latest, 'fitness_best')):.2f}")
        print(f"genome_variance_mean: {latest[0]['genome_variance_mean']:.3f} -> "
              f"{latest[-1]['genome_variance_mean']:.3f}")
        print(f"cma_sigma: {latest[0]['cma_sigma']:.3f} -> {latest[-1]['cma_sigma']:.3f}")
        print(f"seconds/gen (mean): {mean(col(latest, 'seconds')):.1f}")


if __name__ == "__main__":
    main()