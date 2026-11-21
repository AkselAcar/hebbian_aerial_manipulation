#!/usr/bin/env python3
"""How the flights change along training  (plain python3 + matplotlib).

    python3 plot_generations.py --replays results/<run>/replay_theta_gen0000_* results/<run>/replay_theta_gen0100_* ... \
        --out gens_<run>

Takes replays of the same run at several generations (jobs/gen_replays.sh: same seed, so the same
start/goal pairs for every generation) and draws, one curve per generation (median over the flights):
    payload position error [m], payload attitude error [deg], mean |acceleration command| of the 3 drones,
    tilt of the drones [deg], share of swarms still flying, and the evaluation score vs generation.
The generation is read from the folder name (theta_gen<N> or best_gen<N>).
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

GRID, INK = "#dcdbd7", "#3a3a38"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False, "savefig.dpi": 180, "savefig.bbox": "tight"})


def gen_of(folder):
    m = re.search(r"(?:theta_gen|best_gen)(\d+)", os.path.basename(os.path.normpath(folder)))
    return int(m.group(1)) if m else -1


def extra(folder):
    """|acc command| and drone tilt, (flights, T) with NaN after the crash."""
    acc, tilt = [], []
    for f in sorted(f for f in glob.glob(os.path.join(folder, "ep*.npz")) if os.path.basename(f)[2:-4].isdigit()):
        d = np.load(f)
        al = d["alive"].astype(bool)
        a = np.linalg.norm(d["actions"][..., :3], axis=-1).mean(-1)                 # (T, envs)
        if "drone_quat" in d.files:                                                  # x, y, z, w
            q = d["drone_quat"]
            zz = 1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2)                           # R[2, 2]
            tl = np.degrees(np.arccos(np.clip(zz, -1, 1))).max(-1)                   # most tilted drone
        else:
            tl = np.full(a.shape, np.nan)
        a[~al], tl[~al] = np.nan, np.nan
        acc.append(a.T); tilt.append(tl.T)
    return np.concatenate(acc), np.concatenate(tilt)


def curve(fl, key):
    T = fl[0]["T"]
    M = np.full((len(fl), T), np.nan)
    for i, f in enumerate(fl):
        v = f[key]
        M[i, :f["n"]] = np.linalg.norm(v, axis=1) if v.ndim == 2 else v
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replays", nargs="+", required=True)
    p.add_argument("--out", default="gens")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(((gen_of(r), r) for r in a.replays), key=lambda x: x[0])
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax = ax.ravel()
    rows = []
    for i, (g, folder) in enumerate(runs):
        fl, _ = load(folder)
        dt, T = fl[0]["dt"], fl[0]["T"]
        t = np.arange(T) * dt
        col = cmap(i / max(len(runs) - 1, 1))
        acc, tilt = extra(folder)
        with np.errstate(all="ignore"):
            for k, M in enumerate([curve(fl, "pos"), curve(fl, "att"), acc, tilt]):
                ax[k].plot(t, np.nanmedian(M, 0), color=col, lw=1.6, label=f"gen {g}")
        alive = np.array([np.arange(T) < f["n"] for f in fl]).mean(0)
        ax[4].plot(t, 100 * alive, color=col, lw=1.6)
        m = eval_metrics(fl)
        rows.append([g, folder, *(round(m[k], 2) for k in ("score", "success_pct", "survival_pct"))])
    for k, lab in enumerate(["payload position error [m]", "payload attitude error [deg]",
                             "|acceleration command| [m/s²]", "tilt of the most tilted drone [deg]",
                             "swarms still flying [%]"]):
        ax[k].set_ylabel(lab); ax[k].set_xlabel("time [s]")
    ax[0].legend(fontsize=9, ncol=2)
    gs = [r[0] for r in rows]
    ax[5].plot(gs, [r[2] for r in rows], "o-", color=INK, label="evaluation score")
    ax[5].plot(gs, [r[3] for r in rows], "s--", color="#2a78d6", label="success %")
    ax[5].plot(gs, [r[4] for r in rows], "^:", color="#eb6834", label="survival %")
    ax[5].set_xlabel("generation"); ax[5].set_ylim(0, 100); ax[5].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "generations.png")); plt.close(fig)
    with open(os.path.join(a.out, "generations.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["generation", "replay", "eval_score", "success_pct", "survival_pct"]); w.writerows(rows)
    for r in rows:
        print(f"gen {r[0]:4d}: score {r[2]:5.1f} | success {r[3]:5.1f}% | survival {r[4]:5.1f}%")
    print(f"saved in {a.out}/")


if __name__ == "__main__":
    main()