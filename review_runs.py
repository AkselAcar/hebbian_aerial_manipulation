#!/usr/bin/env python3
"""Learning curves and end-of-run numbers of several training runs, from their generation logs
(plain python3 + matplotlib).

    python3 review_runs.py                                   # every logs/generation_log_*_A1_* and *_A2_*
    python3 review_runs.py --logs logs/generation_log_X.csv logs/generation_log_Y.csv
    python3 review_runs.py --ref logs/generation_log_3219114_<tag>.csv   # also draw the run we warm-started from
-> review_runs.png + a table in the terminal (and review_runs.csv).

Fitness is shown as % of the best possible on the task the run was trained on in that generation
(A2 logs that maximum; otherwise 100 per drone for 2000 steps). During a curriculum the task is
easier, and runs with other reward weights are scored differently, so compare A2 runs with A1 only
by their trend here; compare the controllers themselves with the replays (diagnose_replay.py).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#c2489a", "#8a6d1f", "#2fa3b8", "#4a4a48"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d9d8d4",
                     "font.size": 11, "axes.titlesize": 12, "legend.frameon": False})


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def col(name, default=np.nan):
        return np.array([float(r[name]) if r.get(name) not in (None, "") else default for r in rows])
    d = {k: col(k) for k in ("generation", "fitness_mean", "fitness_best", "frac_full_episode", "seconds",
                             "es_lr_now", "fitness_max_per_drone", "steps_simulated", "curr_heading_change_max_deg",
                             "curr_tilt_max_deg", "curr_z_half_range", "pos_weight_now", "collision_dist_now",
                             "pos_err_end_full", "att_err_end_deg_full")}
    fmax = d["fitness_max_per_drone"]
    if np.all(np.isnan(fmax)):                     # older logs: max 100 per drone for 2000 steps
        fmax = 100.0 * d["steps_simulated"] / 2000.0 if not np.all(np.isnan(d["steps_simulated"])) else np.full(len(rows), 100.0)
    d["pct"] = 100 * d["fitness_mean"] / fmax
    d["pct_best"] = 100 * d["fitness_best"] / fmax
    return d


def smooth(x, k=10):
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, (k - 1, 0), mode="edge"), np.ones(k) / k, mode="valid")


def curriculum_end(d):
    """First generation at which every curriculum value has reached its final value (None if no curriculum)."""
    ends = []
    for k in ("curr_heading_change_max_deg", "curr_tilt_max_deg", "curr_z_half_range", "pos_weight_now", "collision_dist_now"):
        v = d[k]
        if np.all(np.isnan(v)) or np.nanmax(v) - np.nanmin(v) < 1e-9:
            continue
        ends.append(int(np.argmax(np.isclose(v, v[-1]))))
    return max(ends) if ends else None


def name_of(path):
    m = re.search(r"generation_log_(.+)\.csv$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="*", default=None)
    p.add_argument("--ref", default=None, help="log of the run the warm starts came from (drawn in grey)")
    p.add_argument("--last", type=int, default=20, help="end-of-run numbers = mean over the last N generations")
    p.add_argument("--out", default="review_runs.png")
    a = p.parse_args()
    logs = a.logs or sorted(glob.glob("logs/generation_log_*_A1_*.csv") + glob.glob("logs/generation_log_*_A2_*.csv"))
    if not logs:
        raise SystemExit("no logs found")

    data = [read(path) for path in logs]
    has_err = any(not np.all(np.isnan(d["att_err_end_deg_full"])) for d in data)   # logs from train_hebbian v8.1 on
    fig, axs = plt.subplots(2 if has_err else 1, 3, figsize=(17, 9.4 if has_err else 4.8), squeeze=False)
    ax = axs[0]
    rows = []
    if a.ref:
        r = read(a.ref)
        ax[0].plot(r["generation"], smooth(r["pct"]), color="#b5b4af", lw=1.5, label=f"{name_of(a.ref)} (start point)")
    for k, path in enumerate(logs):
        d, nm, c = data[k], name_of(path), COLORS[k % len(COLORS)]
        if not len(d["generation"]):
            print(f"{nm}: empty log (crashed or not started), skipped")
            continue
        g = d["generation"]
        ax[0].plot(g, smooth(d["pct"]), color=c, lw=1.8, label=nm)
        ax[1].plot(g, smooth(100 * d["frac_full_episode"]), color=c, lw=1.8)
        ax[2].plot(g, d["es_lr_now"], color=c, lw=1.5)
        ce = curriculum_end(d)
        if ce is not None:
            ax[0].axvline(ce, color=c, ls=":", lw=1.2)
        if has_err and not np.all(np.isnan(d["att_err_end_deg_full"])):
            axs[1][0].plot(g, smooth(d["pos_err_end_full"]), color=c, lw=1.8, label=nm)
            axs[1][1].plot(g, smooth(d["att_err_end_deg_full"]), color=c, lw=1.8, label=nm)
            if ce is not None:
                axs[1][1].axvline(ce, color=c, ls=":", lw=1.2)
        n = min(a.last, len(g))
        rows.append([nm, len(g), round(np.nansum(d["seconds"]) / 3600, 1), round(np.nanmean(d["seconds"]), 0),
                     round(np.nanmean(d["pct"][:n]), 1), round(np.nanmean(d["pct"][-n:]), 1),
                     round(100 * np.nanmean(d["frac_full_episode"][-n:]), 1), "" if ce is None else ce])
    ax[0].set_title("Fitness mean, % of the best possible on that generation's task", loc="left")
    ax[1].set_title("Flights lasting the full 20 s (%)", loc="left")
    ax[2].set_title("OpenAI-ES learning rate", loc="left")
    for i in range(3):
        ax[i].set_xlabel("generation"); ax[i].grid(color="#e6e5e1", lw=0.6)
    ax[0].set_ylabel("%"); ax[1].set_ylabel("%"); ax[1].set_ylim(0, 105)
    ax[0].legend(fontsize=8, loc="lower right")
    if has_err:
        axs[1][0].set_title("Payload-goal distance at the end, full flights (m)", loc="left")
        axs[1][1].set_title("Payload attitude error at the end, full flights (deg)", loc="left")
        axs[1][2].axis("off")
        axs[1][2].text(0, 0.9, "Errors on that generation's task:\nduring a curriculum the task is easier,\n"
                       "so compare runs after the dotted line.", fontsize=10, color="#4a4a48", va="top")
        for i in range(2):
            axs[1][i].set_xlabel("generation"); axs[1][i].set_ylim(bottom=0); axs[1][i].grid(color="#e6e5e1", lw=0.6)
    fig.text(0.01, -0.03, f"curves smoothed over 10 generations; dotted: curriculum finished (full task from there on)",
             fontsize=9, color="#6a6a66")
    fig.tight_layout(); fig.savefig(a.out, dpi=170, bbox_inches="tight")

    head = ["run", "gens", "hours", "s/gen", f"fit% first {a.last}", f"fit% last {a.last}", "full% last", "curr. ends"]
    print(" | ".join(head))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()