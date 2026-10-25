#!/usr/bin/env python3
"""How well does the swarm bring the payload to its goal?  (plain python3 + matplotlib)

Reads the ep*.npz files written by replay_hebbian.py. Only the payload has a goal
(a position + orientation drawn once per episode); the drones have no goal of their
own, they only hold the payload through the cables.

1) Overview figure, all swarms of a replay folder:
       python3 plot_goal_tracking.py --replay results/<run>/replay_theta_<stamp>
   -> <replay>/goal_tracking.png with
      a. top view: for every swarm, payload path from start (o) towards its goal (*),
         end (x); a dotted line joins each end to its goal = the remaining error
      b. side view (x, z), same
      c. distance payload-goal over time: every swarm (thin) and the mean (thick)
      d. end position vs goal, per axis: on the diagonal = goal reached;
         the fitted slope says how much the end follows the goal (1 = perfect, 0 = ignores it)

   and <replay>/goal_tracking_3d.png: the same payload paths and goals in 3D.
   The terminal lists the swarms that end closest to their goal (good examples for 2 and 3).

2) 3D still of one swarm: payload path (coloured by time, marks at 0/1/2/4 s),
   start, end, goal, remaining error, and the 3 drones + cables at the end only, like the
   last frame of the animation (--no_drones hides them):
       python3 plot_goal_tracking.py --replay <folder> --swarm3d --ep 0 --env 5
   -> <replay>/swarm3d_ep0_env5.png

3) Animation of one swarm (drones, cables, payload, goal), for slides:
       python3 plot_goal_tracking.py --replay <folder> --animate --ep 0 --env 0
   -> <replay>/anim_ep0_env0.mp4 (or .gif if ffmpeg is missing)
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_DRONES = ["#2a78d6", "#eb6834", "#1baf7a"]
C_LOAD, C_GOAL, INK, GRID = "#6250d6", "#e34948", "#4a4a48", "#d9d8d4"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False})


def load_swarms(folder):
    """List of dicts, one per swarm: load/goal/drones up to its first crash (or full episode)."""
    files = sorted(glob.glob(os.path.join(folder, "ep*.npz")))
    files = [f for f in files if os.path.basename(f)[2:-4].isdigit()]
    if not files:
        raise SystemExit(f"no ep*.npz in {folder}")
    swarms = []
    for f in files:
        d = np.load(f)
        names = list(d["reason_names"])
        T, n_env = d["alive"].shape
        for e in range(n_env):
            n = int(d["alive"][:, e].sum())
            if n < 2:
                continue
            r = "+".join(nm for nm, x in zip(names, d["reasons"][e]) if x) or "full episode"
            swarms.append(dict(ep=int(os.path.basename(f)[2:-4]), env=e, n=n, T=T, reason=r,
                               load=d["load"][:n, e], goal=d["goal"][:n, e], drones=d["drones"][:n, e],
                               dt=float(d["step_dt"])))
    return swarms


def overview(swarms, folder, max_paths, elev=25, azim=-60):
    full = [s for s in swarms if s["n"] >= s["T"]]
    dt, T = swarms[0]["dt"], swarms[0]["T"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"Payload vs goal: {len(swarms)} swarms ({len(full)} flew the full {T * dt:.0f} s)",
                 x=0.02, ha="left", fontsize=13)
    cmap = plt.get_cmap("tab10")

    # a/b. paths towards the goal (top and side view)
    for a, (i, j, lab) in zip([ax[0, 0], ax[0, 1]], [(0, 1, "y"), (0, 2, "z")]):
        for k, s in enumerate(full[:max_paths]):
            c = cmap(k % 10)
            L, G = s["load"], s["goal"][0]
            a.plot(L[:, i], L[:, j], color=c, lw=1.2, alpha=0.85)
            a.plot(L[0, i], L[0, j], "o", color=c, ms=5)
            a.plot(L[-1, i], L[-1, j], "x", color=c, ms=7, mew=2)
            a.plot(G[i], G[j], "*", color=c, ms=13, mec="white", mew=0.5)
            a.plot([L[-1, i], G[i]], [L[-1, j], G[j]], ":", color=c, lw=1)
        a.set_xlabel("x [m]"); a.set_ylabel(f"{lab} [m]"); a.grid(color=GRID, lw=0.6)
        a.set_aspect("equal", adjustable="datalim")
    ax[0, 0].set_title(f"Top view, {min(max_paths, len(full))} swarms: o start, x end, * goal, "
                       f"dotted = remaining error", loc="left")
    ax[0, 1].set_title("Side view (x, z), same swarms", loc="left")

    # c. distance to goal over time
    a = ax[1, 0]
    D = np.full((len(full), T), np.nan)
    for k, s in enumerate(full):
        D[k] = np.linalg.norm(s["load"] - s["goal"], axis=1)
        a.plot(np.arange(T) * dt, D[k], color=INK, lw=0.6, alpha=0.25)
    if full:
        a.plot(np.arange(T) * dt, np.nanmean(D, 0), color=C_LOAD, lw=2.5, label="mean")
    a.axhline(0.3, ls="--", color=C_GOAL, lw=1.2, label="goal reached (0.3 m, env setting)")
    a.set_xlabel("time [s]"); a.set_ylabel("|payload - goal| [m]"); a.set_ylim(bottom=0)
    a.set_title("Distance payload-goal, full-episode swarms", loc="left"); a.legend(loc="upper right")
    a.grid(color=GRID, lw=0.6)

    # d. end position vs goal, per axis
    a = ax[1, 1]
    lims = [np.inf, -np.inf]
    print(f"{len(full)} full-episode swarms")
    for i, (lab, c) in enumerate(zip("xyz", ["#2a78d6", "#eb6834", "#1baf7a"])):
        g = np.array([s["goal"][0, i] for s in full]); e = np.array([s["load"][-1, i] for s in full])
        st = np.array([s["load"][0, i] for s in full])
        if len(g) < 3:
            continue
        # slope of end position vs goal: 1 = the end follows the goal exactly, 0 = the end ignores the goal.
        # (Not "movement vs requested movement": the swarm forgets its start, which would inflate that.)
        slope = np.polyfit(g, e, 1)[0]
        a.scatter(g, e, s=18, color=c, alpha=0.75, label=f"{lab}: end follows goal at {slope:.0%}")
        lims = [min(lims[0], g.min(), e.min()), max(lims[1], g.max(), e.max())]
        print(f"  {lab}: end follows goal at {slope:.0%} (slope)   corr(end, goal) {np.corrcoef(e, g)[0, 1]:+.2f}"
              f"   corr(end, start) {np.corrcoef(e, st)[0, 1]:+.2f}")
    if np.isfinite(lims[0]):
        a.plot(lims, lims, "--", color=INK, lw=1, label="perfect (end = goal)")
    a.set_xlabel("goal coordinate [m]"); a.set_ylabel("payload end coordinate [m]")
    a.set_title("End position vs goal, per axis", loc="left"); a.legend(loc="upper left")
    a.grid(color=GRID, lw=0.6); a.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(folder, "goal_tracking.png")
    fig.savefig(out, dpi=150)
    if full:
        print(f"  mean distance: start {np.nanmean(D[:, 0]):.2f} m, end {np.nanmean(D[:, -1]):.2f} m, "
              f"within 0.3 m at the end: {np.mean(D[:, -1] < 0.3):.0%}")
        order = np.argsort(-(D[:, 0] - D[:, -1]))[:5]
        print("  swarms that moved the most towards their goal (good examples for --swarm3d / --animate):")
        for k in order:
            print(f"    --ep {full[k]['ep']} --env {full[k]['env']}: {D[k, 0]:.2f} m -> {D[k, -1]:.2f} m")
    print(f"saved {out}")
    if full:
        overview3d(full, folder, max_paths, elev, azim)


def equal_3d(a, pts, pad=0.25, zmin0=True):
    """Same scale on x, y, z so distances look right."""
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    if zmin0:
        lo[2] = max(0.0, lo[2])
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
    a.set_box_aspect(hi - lo)
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z [m]")


def overview3d(full, folder, max_paths, elev, azim):
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(10, 8.5))
    a = fig.add_subplot(projection="3d")
    pts = []
    for k, s in enumerate(full[:max_paths]):
        c = cmap(k % 10)
        L, G = s["load"], s["goal"][0]
        a.plot(L[:, 0], L[:, 1], L[:, 2], color=c, lw=1.4)
        a.scatter(*L[0], color=c, s=25, marker="o")
        a.scatter(*L[-1], color=c, s=45, marker="x", linewidths=2)
        a.scatter(*G, color=c, s=160, marker="*", edgecolors="white", linewidths=0.5)
        a.plot([L[-1, 0], G[0]], [L[-1, 1], G[1]], [L[-1, 2], G[2]], ":", color=c, lw=1.2)
        pts += [L, G[None]]
    equal_3d(a, np.concatenate(pts))
    a.view_init(elev=elev, azim=azim)
    a.set_title(f"Payload paths of {min(max_paths, len(full))} swarms in 3D\n"
                "o start, x end after 20 s, * goal, dotted = remaining error", loc="left")
    fig.tight_layout()
    out = os.path.join(folder, "goal_tracking_3d.png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def swarm3d(swarms, folder, ep, env, elev, azim, show_drones=False):
    s = next((s for s in swarms if s["ep"] == ep and s["env"] == env), None)
    if s is None:
        raise SystemExit(f"no swarm ep {ep} env {env}")
    L, G, Dr, dt, n = s["load"], s["goal"][0], s["drones"], s["dt"], s["n"]
    dist = np.linalg.norm(L - G, axis=1)

    fig = plt.figure(figsize=(14, 7))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    times = [t for t in (0.0, 1.0, 2.0, 4.0) if t / dt < n - 1] + [(n - 1) * dt]
    pts = [L, G[None]]
    # payload path, coloured by time (light = start, dark = end)
    cm = plt.get_cmap("Purples")
    for k in range(0, n - 1, 5):
        k2 = min(k + 5, n - 1)
        a.plot(L[k:k2 + 1, 0], L[k:k2 + 1, 1], L[k:k2 + 1, 2], color=cm(0.35 + 0.65 * k / n), lw=2.4)
    # shadow on the floor of the plot, helps to read depth
    z0 = max(0.0, min(L[:, 2].min(), G[2]) - 0.25)
    a.plot(L[:, 0], L[:, 1], np.full(n, z0), color=GRID, lw=1.5)
    a.scatter(G[0], G[1], z0, marker="*", s=80, color=GRID)
    # time marks along the path
    for t in times:
        k = int(round(t / dt))
        a.scatter(*L[k], color=cm(0.35 + 0.65 * k / n), s=30, edgecolors="white", linewidths=0.5)
        a.text(L[k, 0], L[k, 1], L[k, 2] + 0.06, f"{t:.0f} s", fontsize=8, color=INK)
    a.scatter(*L[0], color=C_LOAD, s=60, marker="o", label="payload start")
    a.scatter(*L[-1], color=C_LOAD, s=70, marker="X", label=f"payload after {(n - 1) * dt:.0f} s")
    a.scatter(*G, marker="*", s=380, color=C_GOAL, edgecolors="white", linewidths=0.6, label="payload goal", zorder=5)
    a.plot([L[-1, 0], G[0]], [L[-1, 1], G[1]], [L[-1, 2], G[2]], ":", color=C_GOAL, lw=1.5,
           label=f"remaining error {dist[-1]:.2f} m")
    if show_drones:   # the 3 drones and cables, final moment only (as in the animation)
        for i in range(3):
            a.plot([Dr[-1, i, 0], L[-1, 0]], [Dr[-1, i, 1], L[-1, 1]], [Dr[-1, i, 2], L[-1, 2]], color=INK, lw=0.8)
            a.scatter(*Dr[-1, i], color=C_DRONES[i], s=35)
        pts.append(Dr[-1])
    equal_3d(a, np.concatenate(pts + [np.array([[L[0, 0], L[0, 1], z0]])]), zmin0=False)
    a.view_init(elev=elev, azim=azim)
    a.legend(loc="upper left", fontsize=8)
    a.set_title(f"Episode {ep}, swarm {env}: payload path ({s['reason']}, {n * dt:.0f} s)", loc="left")

    b = fig.add_subplot(1, 2, 2)
    b.plot(np.arange(n) * dt, dist, color=C_LOAD, lw=1.8)
    b.axhline(0.3, ls="--", color=C_GOAL, lw=1.2, label="goal reached (0.3 m)")
    for t in times:
        b.axvline(t, color=GRID, lw=0.8)
    b.set_xlabel("time [s]"); b.set_ylabel("|payload - goal| [m]"); b.set_ylim(bottom=0)
    b.set_title(f"Distance payload-goal: {dist[0]:.2f} m -> {dist[-1]:.2f} m", loc="left")
    b.legend(loc="upper right"); b.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    out = os.path.join(folder, f"swarm3d_ep{ep}_env{env}.png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


def animate(swarms, folder, ep, env, every, fps):
    from matplotlib import animation
    s = next((s for s in swarms if s["ep"] == ep and s["env"] == env), None)
    if s is None:
        raise SystemExit(f"no swarm ep {ep} env {env}")
    L, G, Dr, dt = s["load"][::every], s["goal"][0], s["drones"][::every], s["dt"] * every
    dist = np.linalg.norm(s["load"] - G, axis=1)
    t_all = np.arange(s["n"]) * s["dt"]

    fig = plt.figure(figsize=(13, 6))
    a3 = fig.add_subplot(1, 2, 1, projection="3d")
    ad = fig.add_subplot(1, 2, 2)
    pts = np.concatenate([L, Dr.reshape(-1, 3), G[None]])
    c, r = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 0.3
    a3.set_xlim(c[0] - r, c[0] + r); a3.set_ylim(c[1] - r, c[1] + r); a3.set_zlim(max(0, c[2] - r), c[2] + r)
    a3.set_xlabel("x [m]"); a3.set_ylabel("y [m]"); a3.set_zlabel("z [m]")
    a3.scatter(*G, marker="*", s=250, color=C_GOAL, label="payload goal")
    trail, = a3.plot([], [], [], color=C_LOAD, lw=1.2, alpha=0.6)
    load_pt, = a3.plot([], [], [], "s", color=C_LOAD, ms=9, label="payload")
    drone_pts = [a3.plot([], [], [], "o", color=C_DRONES[i], ms=7, label=f"drone {i + 1}")[0] for i in range(3)]
    cables = [a3.plot([], [], [], "-", color=INK, lw=0.8)[0] for _ in range(3)]
    err_line, = a3.plot([], [], [], ":", color=C_GOAL, lw=1.2)
    a3.legend(loc="upper left")
    a3.set_title(f"Episode {ep}, swarm {env} ({s['reason']})", loc="left")

    ad.plot(t_all, dist, color=C_LOAD, lw=1.8)
    ad.axhline(0.3, ls="--", color=C_GOAL, lw=1.2, label="goal reached (0.3 m)")
    cursor = ad.axvline(0, color=INK, lw=1)
    ad.set_xlabel("time [s]"); ad.set_ylabel("|payload - goal| [m]"); ad.set_ylim(bottom=0)
    ad.set_title("Distance payload-goal", loc="left"); ad.legend(loc="upper right"); ad.grid(color=GRID, lw=0.6)
    txt = ad.text(0.98, 0.05, "", transform=ad.transAxes, ha="right", fontsize=11)

    def update(k):
        trail.set_data_3d(L[:k + 1, 0], L[:k + 1, 1], L[:k + 1, 2])
        load_pt.set_data_3d([L[k, 0]], [L[k, 1]], [L[k, 2]])
        for i in range(3):
            drone_pts[i].set_data_3d([Dr[k, i, 0]], [Dr[k, i, 1]], [Dr[k, i, 2]])
            cables[i].set_data_3d([Dr[k, i, 0], L[k, 0]], [Dr[k, i, 1], L[k, 1]], [Dr[k, i, 2], L[k, 2]])
        err_line.set_data_3d([L[k, 0], G[0]], [L[k, 1], G[1]], [L[k, 2], G[2]])
        cursor.set_xdata([k * dt, k * dt])
        txt.set_text(f"t = {k * dt:4.1f} s   distance {np.linalg.norm(L[k] - G):.2f} m")
        a3.view_init(elev=22, azim=-60 + 0.05 * k)  # slow rotation helps reading depth
        return [trail, load_pt, err_line, cursor, txt, *drone_pts, *cables]

    anim = animation.FuncAnimation(fig, update, frames=len(L), interval=1000 / fps)
    fig.tight_layout()
    out = os.path.join(folder, f"anim_ep{ep}_env{env}.mp4")
    try:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=110)
    except Exception as ex:
        print(f"ffmpeg not usable ({ex.__class__.__name__}), writing a gif instead")
        out = out.replace(".mp4", ".gif")
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=80)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True, help="replay folder written by replay_hebbian.py")
    p.add_argument("--max_paths", type=int, default=10, help="swarms drawn in the top/side views")
    p.add_argument("--animate", action="store_true", help="make an animation of one swarm instead")
    p.add_argument("--ep", type=int, default=0)
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--every", type=int, default=4, help="animation: keep every k-th step (4 -> 25 fps = real time)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--swarm3d", action="store_true", help="3D still of one swarm (--ep, --env)")
    p.add_argument("--no_drones", action="store_true", help="--swarm3d: hide the 3 drones and cables (shown at the end only)")
    p.add_argument("--elev", type=float, default=25, help="3D view: camera height angle [deg]")
    p.add_argument("--azim", type=float, default=-60, help="3D view: camera rotation [deg]")
    args = p.parse_args()
    swarms = load_swarms(args.replay)
    if args.animate:
        animate(swarms, args.replay, args.ep, args.env, args.every, args.fps)
    elif args.swarm3d:
        swarm3d(swarms, args.replay, args.ep, args.env, args.elev, args.azim, not args.no_drones)
    else:
        overview(swarms, args.replay, args.max_paths, args.elev, args.azim)


if __name__ == "__main__":
    main()  