#!/usr/bin/env python3
"""Payload-mass robustness of a trained controller  (plain python3 + matplotlib).

Reads the replay folders written by jobs/mass_test.sh, i.e.
    <run>/replay_theta_m<mass>_<stamp>/          plasticity on  (Hebbian lr 0.001)
    <run>/replay_theta_m<mass>_heblr0_<stamp>/   plasticity off (weights frozen at their evolved start values)
and plots, against the payload mass:
    a. share of flights that last the full 20 s without crashing
    b. distance payload-goal at the end of the full flights (mean, with the 25-75 % band)
    c. share of full flights ending within 0.3 m of the goal
If a mass was replayed more than once, the newest folder is used.

    python3 plot_mass_test.py --run results/3219114_s0.05_eslr0.01_BEST
    python3 plot_mass_test.py --run <run> --train_mass 1.4          # mark the training mass
    python3 plot_mass_test.py --run <run> --train_range 1.0 1.8     # or shade a training range
-> <run>/mass_test.png, and a table in the terminal.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_ON, C_OFF, INK, GRID, C_GOAL = "#2a78d6", "#eb6834", "#4a4a48", "#d9d8d4", "#e34948"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False})
NAME = re.compile(r"replay_theta_m([0-9.]+)(_heblr([0-9.e-]+))?_(\d{8}_\d{6})$")


def collect(run):
    """{(mass, plastic): folder}, newest folder per key."""
    found = {}
    for d in sorted(glob.glob(os.path.join(run, "replay_theta_m*"))):
        m = NAME.search(os.path.basename(d))
        if not m or not glob.glob(os.path.join(d, "ep*.npz")):
            continue
        mass = float(m.group(1))
        plastic = m.group(3) is None or float(m.group(3)) > 0
        key = (mass, plastic)
        if key not in found or m.group(4) > found[key][0]:
            found[key] = (m.group(4), d)
    return {k: v[1] for k, v in found.items()}


def stats(folder):
    full, ends, n = [], [], 0
    for f in glob.glob(os.path.join(folder, "ep*.npz")):
        if not os.path.basename(f)[2:-4].isdigit():
            continue
        d = np.load(f)
        T = d["alive"].shape[0]
        ok = d["first_crash"] >= T
        n += ok.size
        full.append(ok)
        if ok.any():
            ends.append(np.linalg.norm(d["load"][-1][ok] - d["goal"][-1][ok], axis=1))
    full = np.concatenate(full)
    ends = np.concatenate(ends) if ends else np.array([np.nan])
    return dict(n=n, full=full.mean(), end=np.nanmean(ends), q25=np.nanpercentile(ends, 25),
                q75=np.nanpercentile(ends, 75), within=np.nanmean(ends < 0.3) if ends.size else np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="results folder of the trained run")
    p.add_argument("--train_mass", type=float, default=1.4, help="mass used in training (dashed line)")
    p.add_argument("--train_range", type=float, nargs=2, default=None, help="mass range used in training (shaded)")
    args = p.parse_args()

    found = collect(args.run)
    if not found:
        raise SystemExit(f"no replay_theta_m* folders with data in {args.run}")
    res = {k: stats(v) for k, v in found.items()}

    print(f"{'mass':>5} {'plasticity':>10} {'flights':>7} {'full 20 s':>9} {'end dist':>8} {'<0.3 m':>7}")
    for (mass, plastic), s in sorted(res.items()):
        print(f"{mass:5.2f} {'on' if plastic else 'off':>10} {s['n']:7d} {s['full']:9.0%} "
              f"{s['end']:7.2f}m {s['within']:7.0%}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Payload-mass test of {os.path.basename(os.path.normpath(args.run))} (no retraining)",
                 x=0.01, ha="left", fontsize=13)
    for plastic, c, lab in [(True, C_ON, "plasticity on"), (False, C_OFF, "plasticity off (weights frozen)")]:
        keys = sorted(k for k in res if k[1] == plastic)
        if not keys:
            continue
        m = np.array([k[0] for k in keys])
        S = [res[k] for k in keys]
        ax[0].plot(m, [100 * s["full"] for s in S], "o-", color=c, lw=2, label=lab)
        ax[1].plot(m, [s["end"] for s in S], "o-", color=c, lw=2, label=lab)
        ax[1].fill_between(m, [s["q25"] for s in S], [s["q75"] for s in S], color=c, alpha=0.15, lw=0)
        ax[2].plot(m, [100 * s["within"] for s in S], "o-", color=c, lw=2, label=lab)
    for a in ax:
        if args.train_range:
            a.axvspan(*args.train_range, color=GRID, alpha=0.5, lw=0, label="training range")
        else:
            a.axvline(args.train_mass, ls="--", color=INK, lw=1, label=f"training mass {args.train_mass:g} kg")
        a.set_xlabel("payload mass [kg]"); a.grid(color=GRID, lw=0.6)
    ax[0].set_ylabel("%"); ax[0].set_ylim(0, 105); ax[0].set_title("Flights lasting 20 s without crash", loc="left")
    ax[1].set_ylabel("m"); ax[1].set_ylim(bottom=0)
    ax[1].set_title("Payload-goal distance at the end (band: 25-75 %)", loc="left")
    ax[1].axhline(0.3, ls=":", color=C_GOAL, lw=1.2)
    ax[2].set_ylabel("%"); ax[2].set_ylim(bottom=0); ax[2].set_title("Full flights ending within 0.3 m", loc="left")
    ax[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(args.run, "mass_test.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()