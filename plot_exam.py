#!/usr/bin/env python3
"""Exam plots: every controller on the same flights, side by side  (plain python3 + matplotlib).

    python3 plot_exam.py --replays <replay folder> <replay folder> ... --out exam_plots

Each replay folder is one controller's exam (jobs/review_replays.sh, same seed, normal task).
Draws:
  exam_bars.png       evaluation score, success %, full flights % per controller
  exam_over_time.png  median payload position error, median attitude error and share of swarms still
                      flying over the 20 s, one line per controller
Colours by family: Hebbian runs of Fuda's recipe (E1/E2/F1) blue, MAPPO orange, older Hebbian runs grey.
Labels = run folder name without the job id (or --labels, same order as --replays).
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_replay import eval_metrics, load

GRID, INK = "#dcdbd7", "#3a3a38"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False, "savefig.dpi": 180, "savefig.bbox": "tight"})
FAMILY = {"new": plt.get_cmap("Blues"), "mappo": plt.get_cmap("Oranges"), "old": plt.get_cmap("Greys")}


def label_of(folder):
    run = os.path.basename(os.path.dirname(os.path.normpath(folder)))
    return re.sub(r"^\d+_", "", run)                      # drop the job id


def family_of(label):
    if label.startswith("mappo"):
        return "mappo"
    if re.match(r"(E[2-9]|E1|F\d)_", label):
        return "new"
    return "old"


def over_time(fl, key):
    T = max(f["T"] for f in fl)
    M = np.full((len(fl), T), np.nan)
    for i, f in enumerate(fl):
        v = f[key]
        M[i, :f["n"]] = np.linalg.norm(v, axis=1) if v.ndim == 2 else v
    with np.errstate(all="ignore"):
        return np.nanmedian(M, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default="exam_plots")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = a.labels or [label_of(r) for r in a.replays]
    runs = []
    for lab, folder in zip(labels, a.replays):
        fl, _ = load(folder)
        runs.append(dict(label=lab, fam=family_of(lab), fl=fl, m=eval_metrics(fl), dt=fl[0]["dt"]))
    runs.sort(key=lambda r: -r["m"]["score"])                 # best first
    # one colour per controller, shades within its family
    count = {k: sum(r["fam"] == k for r in runs) for k in FAMILY}
    seen = {k: 0 for k in FAMILY}
    for r in runs:
        k = r["fam"]; seen[k] += 1
        r["c"] = FAMILY[k](0.9 - 0.45 * (seen[k] - 1) / max(count[k] - 1, 1))

    # --- bars -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 0.45 * len(runs) + 1.8), sharey=True)
    y = np.arange(len(runs))
    for k, (key, title) in enumerate([("score", "evaluation score (0-100)"), ("success_pct", "success %"),
                                      ("survival_pct", "full 20 s flights %")]):
        v = [r["m"][key] for r in runs]
        ax[k].barh(y, v, color=[r["c"] for r in runs], height=0.7)
        for yi, vi in zip(y, v):
            ax[k].text(vi + 1, yi, f"{vi:.0f}", va="center", fontsize=10, color=INK)
        ax[k].set_xlim(0, 105); ax[k].set_title(title); ax[k].grid(axis="y", visible=False)
    ax[0].set_yticks(y); ax[0].set_yticklabels([r["label"] for r in runs]); ax[0].invert_yaxis()
    fig.text(0.01, -0.02, "same 64 flights for every controller; success = never crashed and ends within 0.3 m "
             "and 0.4 rad; score counts 0 after a crash", fontsize=10, color="#6b6b68")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "exam_bars.png")); plt.close(fig)

    # --- over time ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for r in runs:
        fl = r["fl"]; T = max(f["T"] for f in fl); t = np.arange(T) * r["dt"]
        ax[0].plot(t, over_time(fl, "pos"), color=r["c"], lw=1.8, label=r["label"])
        ax[1].plot(t, over_time(fl, "att"), color=r["c"], lw=1.8)
        alive = np.array([np.arange(T) < f["n"] for f in fl]).mean(0)
        ax[2].plot(t, 100 * alive, color=r["c"], lw=1.8)
    for k, lab in enumerate(["payload position error [m] (median)", "payload attitude error [deg] (median)",
                             "swarms still flying [%]"]):
        ax[k].set_xlabel("time [s]"); ax[k].set_ylabel(lab)
    ax[2].set_ylim(0, 102)
    ax[0].legend(fontsize=9, loc="upper right")
    fig.text(0.01, -0.03, "medians over the swarms still flying at that time; blue = Hebbian (Fuda's recipe), "
             "orange = MAPPO, grey = older Hebbian runs", fontsize=10, color="#6b6b68")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "exam_over_time.png")); plt.close(fig)
    for r in runs:
        print(f"{r['label']:28s} score {r['m']['score']:5.1f} | success {r['m']['success_pct']:5.1f}% | "
              f"full {r['m']['survival_pct']:5.1f}%")
    print(f"saved {a.out}/exam_bars.png and {a.out}/exam_over_time.png")


if __name__ == "__main__":
    main()