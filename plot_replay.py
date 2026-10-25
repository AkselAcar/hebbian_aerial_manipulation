#!/usr/bin/env python3
"""Plot one swarm's episode from a replay_hebbian.py recording.

Usage (no Isaac Lab needed, plain python3 + matplotlib):
    python3 plot_replay.py --replay results/<run>/replay_theta_<stamp>
        -> picks the first swarm whose first crash was body_pos_outside
           (or, if none, the first swarm that crashed at all)
    python3 plot_replay.py --replay ... --reason illegal_contact
    python3 plot_replay.py --replay ... --ep 0 --env 5

Saves <replay>/ep<k>_env<e>_<reason>.png with:
    1. top view (x, y) of the 3 drones, the payload and the goal, with the bounding box
    2. side view (x, z)
    3. x, y, z of the payload and the goal over time, with the box limits
    4. distance payload -> goal and lowest/highest drone coordinate over time
    5. commanded linear accelerations (all 3 drones)
    6. commanded body rates (all 3 drones)
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_DRONES = ["#2a78d6", "#eb6834", "#1baf7a"]   # drone 1/2/3
C_LOAD = "#6250d6"                              # payload
C_GOAL = "#e34948"                              # goal
INK, GRID = "#4a4a48", "#d9d8d4"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False})


def pick(data, ep_files, args):
    """Return (ep_index, env_index) to plot."""
    names = list(data[0]["reason_names"])
    for k, d in enumerate(data):
        if args.ep is not None and k != args.ep:
            continue
        T = d["alive"].shape[0]
        for e in range(d["first_crash"].size):
            if args.env is not None:
                if e == args.env:
                    return k, e
                continue
            crashed = d["first_crash"][e] < T or d["reasons"][e].any()
            if args.reason:
                if d["reasons"][e][names.index(args.reason)]:
                    return k, e
            elif crashed and d["reasons"][e][names.index("body_pos_outside")]:
                return k, e
    if args.reason is None and args.env is None:      # fallback: first crash of any kind
        for k, d in enumerate(data):
            for e in range(d["first_crash"].size):
                if d["reasons"][e].any():
                    return k, e
        return 0, 0
    raise SystemExit("no swarm matches the requested --ep/--env/--reason")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True, help="replay folder written by replay_hebbian.py")
    p.add_argument("--ep", type=int, default=None)
    p.add_argument("--env", type=int, default=None)
    p.add_argument("--reason", type=str, default=None, help="e.g. body_pos_outside, illegal_contact")
    args = p.parse_args()

    ep_files = sorted(glob.glob(os.path.join(args.replay, "ep*.npz")),
                      key=lambda f: int(os.path.basename(f)[2:].split(".")[0].split("_")[0]))
    ep_files = [f for f in ep_files if os.path.basename(f)[2:].split(".")[0].isdigit()]
    data = [np.load(f) for f in ep_files]
    k, e = pick(data, ep_files, args)
    d = data[k]
    names = list(d["reason_names"])
    dt, bbox = float(d["step_dt"]), float(d["bbox"])

    live = d["alive"][:, e]                      # True up to and including the crash step
    n = int(live.sum())
    drones = d["drones"][:n, e]                  # (n, 3, 3)
    load, goal, acts = d["load"][:n, e], d["goal"][:n, e], d["actions"][:n, e]
    t = np.arange(n) * dt
    reason = "+".join(nm for nm, f in zip(names, d["reasons"][e]) if f) or "never crashed"
    print(f"episode {k}, env {e}: {n} steps ({n * dt:.1f} s), ended by: {reason}")

    fig, ax = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle(f"Episode {k}, swarm {e}: {n * dt:.1f} s, ended by {reason}", x=0.02, ha="left", fontsize=12)

    # 1. top view
    a = ax[0, 0]
    a.add_patch(plt.Rectangle((-bbox, -bbox), 2 * bbox, 2 * bbox, fill=False, ls="--", ec=INK, lw=1))
    for i in range(3):
        a.plot(drones[:, i, 0], drones[:, i, 1], color=C_DRONES[i], lw=1.5, label=f"drone {i + 1}")
        a.plot(drones[-1, i, 0], drones[-1, i, 1], "x", color=C_DRONES[i], ms=9, mew=2)
    a.plot(load[:, 0], load[:, 1], color=C_LOAD, lw=2, label="payload")
    a.plot(load[0, 0], load[0, 1], "o", color=C_LOAD, ms=7)
    a.plot(goal[0, 0], goal[0, 1], "*", color=C_GOAL, ms=14, label="goal")   # constant within an episode
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_xlim(-bbox * 1.15, bbox * 1.15); a.set_ylim(-bbox * 1.15, bbox * 1.15)
    a.set_title("Top view (o = start, x = end, dashed = bounding box)", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    a.grid(color=GRID, lw=0.6)

    # 2. side view
    a = ax[0, 1]
    a.axvline(-bbox, ls="--", c=INK, lw=1); a.axvline(bbox, ls="--", c=INK, lw=1)
    a.axhline(bbox, ls="--", c=INK, lw=1); a.axhline(0.1, ls=":", c=INK, lw=1)
    for i in range(3):
        a.plot(drones[:, i, 0], drones[:, i, 2], color=C_DRONES[i], lw=1.5)
        a.plot(drones[-1, i, 0], drones[-1, i, 2], "x", color=C_DRONES[i], ms=9, mew=2)
    a.plot(load[:, 0], load[:, 2], color=C_LOAD, lw=2)
    a.plot(goal[0, 0], goal[0, 2], "*", color=C_GOAL, ms=14)
    a.set_xlabel("x [m]"); a.set_ylabel("z [m]")
    a.set_title("Side view (dotted = 10 cm ground limit)", loc="left"); a.grid(color=GRID, lw=0.6)

    # 3. payload and goal over time
    a = ax[1, 0]
    for j, (lab, ls) in enumerate([("x", "-"), ("y", "--"), ("z", ":")]):
        a.plot(t, load[:, j], color=C_LOAD, ls=ls, lw=1.6, label=f"payload {lab}")
        a.plot(t, goal[:, j], color=C_GOAL, ls=ls, lw=1.2, label=f"goal {lab}")
    a.axhline(bbox, c=INK, lw=0.8); a.axhline(-bbox, c=INK, lw=0.8)
    a.set_xlabel("time [s]"); a.set_ylabel("position [m]")
    a.set_title("Payload vs goal (lines at ± bounding box)", loc="left"); a.legend(ncol=3, loc="best")
    a.grid(color=GRID, lw=0.6)

    # 4. distance to goal and how close the drones get to the box
    a = ax[1, 1]
    a.plot(t, np.linalg.norm(load - goal, axis=1), color=C_LOAD, lw=2, label="|payload - goal|")
    a.plot(t, np.abs(drones).max(axis=(1, 2)), color=INK, lw=1.4, label="largest |drone coordinate|")
    a.axhline(bbox, ls="--", c=INK, lw=1)
    a.set_xlabel("time [s]"); a.set_ylabel("m")
    a.set_title("Distance to goal, and distance to the box edge (dashed)", loc="left"); a.legend(loc="best")
    a.grid(color=GRID, lw=0.6)

    # 5./6. commands
    for col, sl, title, unit in [(0, slice(0, 3), "Commanded linear acceleration (x/y/z)", "m/s²"),
                                 (1, slice(3, 5), "Commanded body rates (roll/pitch)", "rad/s")]:
        a = ax[2, col]
        for i in range(3):
            for j, ls in zip(range(sl.start, sl.stop), ["-", "--", ":"]):
                a.plot(t, acts[:, i, j], color=C_DRONES[i], ls=ls, lw=1,
                       label=f"drone {i + 1}" if j == sl.start else None)
        a.set_xlabel("time [s]"); a.set_ylabel(unit); a.set_title(title, loc="left")
        a.legend(loc="best"); a.grid(color=GRID, lw=0.6)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.replay, f"ep{k}_env{e}_{reason.replace('+', '_')}.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")

    # short overview of all swarms in the recording
    print("\nall swarms in this recording:")
    for kk, dd in enumerate(data):
        T = dd["alive"].shape[0]
        for ee in range(dd["first_crash"].size):
            r = "+".join(nm for nm, f in zip(names, dd["reasons"][ee]) if f) or "never crashed"
            print(f"  ep {kk} env {ee:3d}: {dd['first_crash'][ee]:5d} steps | {r}")


if __name__ == "__main__":
    main()