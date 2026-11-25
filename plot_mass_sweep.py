#!/usr/bin/env python3
"""Out-of-distribution test: how each controller copes with payload masses it was never trained on.

    python3 plot_mass_sweep.py --replays results/*/replay_*_m* --out mass_sweep

Takes the replays of jobs/mass_replays.sh (same 64 flights per controller and mass). The controller is read
from the run folder (+ "plasticity off" when the replay ran with Hebbian lr 0), the mass from the replay files.
Draws evaluation score, success % and full flights % against the payload mass, one line per controller,
with the training mass (1.4 kg) marked. Also writes mass_sweep.csv.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_replay import eval_metrics, load

COLORS = ["#2a78d6", "#8fb8ea", "#eb6834", "#3a3a38", "#4a9e5c", "#a05db5"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dcdbd7", "legend.frameon": False, "savefig.dpi": 180, "savefig.bbox": "tight"})


def controller_of(folder):
    run = os.path.basename(os.path.dirname(os.path.normpath(folder)))
    name = re.sub(r"^\d+_", "", run)
    if "heblr0_" in os.path.basename(folder) or os.path.basename(folder).endswith("heblr0"):
        name += " (plasticity off)"
    return name


def mass_of(folder):
    f = sorted(glob.glob(os.path.join(folder, "ep*.npz")))[0]
    d = np.load(f)
    if "payload_mass" in d.files:
        return round(float(d["payload_mass"]), 2)
    m = re.search(r"_m([\d.]+)", os.path.basename(folder))
    return float(m.group(1)) if m else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--train_mass", type=float, default=1.4)
    p.add_argument("--out", default="mass_sweep")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}                                           # controller -> {mass: metrics}
    for folder in a.replays:
        if not glob.glob(os.path.join(folder, "ep*.npz")):
            continue
        fl, _ = load(folder)
        data.setdefault(controller_of(folder), {})[mass_of(folder)] = eval_metrics(fl)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    rows = []
    for k, (ctrl, by_mass) in enumerate(sorted(data.items())):
        ms = sorted(by_mass)
        for j, (key, title) in enumerate([("score", "evaluation score (0-100)"), ("success_pct", "success %"),
                                          ("survival_pct", "full 20 s flights %")]):
            ls = "--" if "plasticity off" in ctrl else "-"
            ax[j].plot(ms, [by_mass[m][key] for m in ms], ls, marker="o", color=COLORS[k % len(COLORS)],
                       lw=2, label=ctrl)
            ax[j].set_title(title); ax[j].set_ylim(0, 102); ax[j].set_xlabel("payload mass [kg]")
        rows += [[ctrl, m, *(round(by_mass[m][x], 1) for x in ("score", "success_pct", "survival_pct"))] for m in ms]
    for j in range(3):
        ax[j].axvline(a.train_mass, color="#8a8a86", lw=1, ls=":")
    ax[0].text(a.train_mass, 3, " trained at 1.4 kg", color="#6b6b68", fontsize=10)
    ax[0].legend(fontsize=10, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "mass_sweep.png")); plt.close(fig)
    with open(os.path.join(a.out, "mass_sweep.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["controller", "mass_kg", "eval_score", "success_pct", "full_pct"]); w.writerows(rows)
    for r in rows:
        print(f"{r[0]:40s} {r[1]:4.1f} kg | score {r[2]:5.1f} | success {r[3]:5.1f}% | full {r[4]:5.1f}%")
    print(f"saved {a.out}/mass_sweep.png and mass_sweep.csv")


if __name__ == "__main__":
    main()