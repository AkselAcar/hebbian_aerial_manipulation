#!/usr/bin/env python3
"""Hover test figure (plain python3 + matplotlib): how the payload tilts when every drone gets action 0.

    python3 plot_hover_tilt.py                       # newest results/*/replay_zero_action_* folder
    python3 plot_hover_tilt.py --replay <folder> --env 0 --times 0 0.5 3
-> <folder>/hover_tilt.png

Left: payload tilt (angle between the payload's up axis and the vertical) over time, every swarm (thin) and the
median (bold), while the swarm is alive.
Right: one swarm seen from the side and from the back at a few instants, in the payload's own heading at t = 0
(x = payload forward, the side where drones 1 and 2 are attached; drone 3 is attached at the back).
Payload drawn as the 0.55 x 0.45 x 0.15 m box of the flycrane asset, cables from the attachment points to the drones.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, GRID, MUTED = "#3d3d3a", "#e4e3df", "#b9b8b2"
C_MED = "#2a78d6"
TIME_COLORS = ["#b9b8b2", "#eb6834", "#2a78d6", "#6250d6"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "font.size": 12, "axes.titlesize": 13, "legend.frameon": False})

# flycrane_rod geometry (Jack's URDF), payload link frame
ATTACH = np.array([[0.27, 0.22, 0.14], [0.27, -0.22, 0.14], [-0.27, 0.0, 0.14]])   # cables 1, 2, 3
BOX_C, BOX_S = np.array([0.0, 0.0, 0.06]), np.array([0.55, 0.45, 0.15])
ODOM = np.array([0.0, 0.0, 0.08])          # the saved "load" position = load_odometry_sensor_link


def tilt_deg(R):
    return np.degrees(np.arccos(np.clip(R[..., 2, 2], -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default=None)
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.5, 3.0])
    a = p.parse_args()
    folder = a.replay or max(glob.glob("results/*/replay_zero_action_*"), key=os.path.getmtime)
    d = np.load(os.path.join(folder, "ep0.npz"))
    R, alive, dt = d["load_R"], d["alive"].astype(bool), float(d["step_dt"])
    load, drones = d["load"], d["drones"]                      # (T, E, 3), (T, E, 3, 3)
    T, E = alive.shape
    t = np.arange(T) * dt
    tilt = tilt_deg(R)
    tilt_a = np.where(alive, tilt, np.nan)
    last = int(np.max(np.where(alive.sum(1) >= E // 2)[0]))  # median shown while >= half the swarms alive

    fig = plt.figure(figsize=(15, 5.2))
    ax0 = fig.add_axes([0.05, 0.13, 0.36, 0.75])
    for e in range(E):
        ax0.plot(t, tilt_a[:, e], color=MUTED, lw=0.6, alpha=0.7)
    ax0.plot(t[:last], np.nanmedian(tilt_a[:last], 1), color=C_MED, lw=2.5, label="median over swarms")
    ax0.plot([], [], color=MUTED, lw=1, label=f"each of the {E} swarms (until it crashes)")
    ax0.set_xlabel("time [s]"); ax0.set_ylabel("payload tilt [deg]")
    ax0.set_title("Zero actions: the payload tilts by itself", loc="left")
    ax0.set_ylim(0, max(60, np.nanmax(tilt_a) + 5)); ax0.grid(color=GRID, lw=0.6)
    ax0.legend(loc="lower right", fontsize=10)
    k3 = min(int(3 / dt), T - 1)
    ax0.annotate(f"{np.nanmedian(tilt_a[k3]):.0f}° at 3 s", (t[k3], np.nanmedian(tilt_a[k3])),
                 xytext=(t[k3] + 1.0, np.nanmedian(tilt_a[k3]) + 8), color=C_MED,
                 arrowprops=dict(arrowstyle="->", color=C_MED))

    # side / back views of one swarm, in the payload heading at t = 0
    e = a.env
    yaw0 = np.arctan2(R[0, e, 1, 0], R[0, e, 0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    views = [("side view (payload forward ->)", 0), ("back view (payload left ->)", 1)]
    corners = np.array(list(itertools.product([-0.5, 0.5], repeat=3))) * BOX_S + BOX_C
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners[i] != corners[j]) == 1]
    for v, (title, axis) in enumerate(views):
        ax = fig.add_axes([0.47 + v * 0.27, 0.13, 0.24, 0.75])
        for n, tt in enumerate(a.times):
            k = min(int(round(tt / dt)), T - 1)
            if not alive[k, e]:
                continue
            col = TIME_COLORS[n % len(TIME_COLORS)]
            Rk = R[k, e]
            origin = load[k, e] - Rk @ ODOM                        # payload link origin
            to_view = lambda P: (Rz @ (P - load[0, e]).T).T          # centred on the start, heading-aligned
            box = to_view(origin + (Rk @ corners.T).T)
            for i, j in edges:
                ax.plot(box[[i, j], axis], box[[i, j], 2], color=col, lw=1.6)
            att = to_view(origin + (Rk @ ATTACH.T).T)
            dr = to_view(drones[k, e])
            for q in range(3):
                ax.plot([att[q, axis], dr[q, axis]], [att[q, 2], dr[q, 2]], color=col, lw=1.0, ls="--")
                ax.plot(dr[q, axis], dr[q, 2], "o", color=col, ms=7)
                if n == len(a.times) - 1 and not (axis == 0 and q == 1):   # drones 1, 2 overlap from the side
                    lab = "drones 1, 2" if (axis == 0 and q == 0) else f"drone {q + 1}"
                    ax.annotate(lab, (dr[q, axis], dr[q, 2]), xytext=(4, 6),
                                textcoords="offset points", fontsize=9, color=INK)
            ax.plot([], [], color=col, lw=2, label=f"t = {tt:g} s, tilt {tilt[k, e]:.0f}°")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, loc="left"); ax.set_xlabel("m"); ax.set_ylabel("height change [m]" if v == 0 else "")
        ax.grid(color=GRID, lw=0.6)
        if v == 0:
            ax.legend(loc="lower left", fontsize=9)
    fig.text(0.47, 0.01, f"swarm {e}; box = payload, dashed = cables, dots = drones", fontsize=9, color="#6a6a66")
    out = os.path.join(folder, "hover_tilt.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()