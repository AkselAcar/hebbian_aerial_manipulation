#!/usr/bin/env python3
"""Error over time for several replays, e.g. normal vs --freeze_after 5, or several runs (--freeze_at 0)
(plain python3 + matplotlib). Full 20 s flights only; line = median over flights, band = 25-75 %.

    python3 plot_freeze.py --replay <normal folder> <freeze folder> --labels "plasticity on" "frozen after 5 s" \
        --freeze_at 5 --out freeze_compare.png
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4a4a48", "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#c2489a", "#8a6d1f", "#2fa3b8"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d9d8d4",
                     "font.size": 12, "axes.titlesize": 13, "legend.frameon": False})


def load(folder):
    pos, att, dt = [], [], 0.01
    for f in sorted(glob.glob(os.path.join(folder, "ep*.npz"))):
        if not os.path.basename(f)[2:-4].isdigit():
            continue
        d = np.load(f)
        T = d["alive"].shape[0]
        ok = d["first_crash"] >= T                                   # full flights only
        if not ok.any():
            continue
        pos.append(np.linalg.norm(d["load"] - d["goal"], axis=-1)[:, ok])   # (T, flights)
        att.append(np.degrees(d["att_err"][:, ok]) if "att_err" in d.files else np.full((T, ok.sum()), np.nan))
        dt = float(d["step_dt"]) if "step_dt" in d.files else dt
    if not pos:
        raise SystemExit(f"no full flights in {folder}")
    return np.concatenate(pos, 1), np.concatenate(att, 1), dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--freeze_at", type=float, default=5.0, help="mark this time with a dashed line (s); 0 = none")
    p.add_argument("--out", default="freeze_compare.png")
    a = p.parse_args()
    labels = a.labels or [os.path.basename(os.path.normpath(r)) for r in a.replay]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for k, (folder, lab) in enumerate(zip(a.replay, labels)):
        pos, att, dt = load(folder)
        t = np.arange(pos.shape[0]) * dt
        c = COLORS[k % len(COLORS)]
        for i, (e, unit) in enumerate([(pos, "m"), (att, "deg")]):
            ax[i].plot(t, np.nanmedian(e, 1), color=c, lw=2, label=f"{lab} ({e.shape[1]} flights)")
            ax[i].fill_between(t, np.nanpercentile(e, 25, 1), np.nanpercentile(e, 75, 1), color=c, alpha=0.15, lw=0)
        print(f"{lab}: position error median at 5/10/20 s = " +
              " / ".join(f"{np.nanmedian(pos[min(int(s / dt), len(t) - 1)]):.2f}" for s in (5, 10, 20)) + " m")
    for i, (title, unit) in enumerate([("Payload-goal distance", "m"), ("Payload attitude error", "deg")]):
        if a.freeze_at > 0:
            ax[i].axvline(a.freeze_at, ls="--", color="#4a4a48", lw=1)
        ax[i].set_title(title, loc="left"); ax[i].set_xlabel("time [s]"); ax[i].set_ylabel(unit)
        ax[i].set_ylim(bottom=0); ax[i].grid(color="#e6e5e1", lw=0.6)
    ax[0].legend(loc="upper right", fontsize=10)
    fig.text(0.01, -0.02, "line: median over full flights, band: 25-75 %" + ("; dashed: freeze time" if a.freeze_at > 0 else ""), fontsize=9, color="#6a6a66")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()