#!/usr/bin/env python3
"""Figures + CSVs for the "Preliminary results" slide  (plain python3 + matplotlib).

Every figure is 6 x 4 inch (two-panel figures 6 x 6.5), fonts >= 14 pt, 300 dpi, no red/green
pairs, and comes with the CSV of the numbers it shows.

    python3 slide_figures.py --replay <replay folder> --out slide_figs
        fig1a_sweep_survival.png  fig1b_sweep_fitness.png   OpenAI-ES sweep 2 (from logs/)
        fig2_errors.png           position + attitude error over time, all flights + example
        fig3_numbers.csv          the numbers of the slide (also printed)
        fig4a_weights_drift.png   how far each drone's weights move from the start weights
        fig4b_weights_between.png how different the three drones' weights become
        fig6_swing.png            example flight: attitude error, payload angular speed, swing angle
    python3 slide_figures.py --replay <replay folder> --out slide_figs --video
        fig5_example_flight.mp4 (+ .gif): 3D re-render with the payload frame and the goal frame

The replay folder must come from jobs/slide_replay.sh (it records the attitude, the payload
velocity and the weights; older replays lack them). The example flight is the full flight with
the median final position error (a typical flight, not the best); change it with --example EP ENV.
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

BLUE, ORANGE, PURPLE, GRAY, INK, GRID = "#2a78d6", "#eb6834", "#6250d6", "#8a8a86", "#3a3a38", "#dcdbd7"
DRONE_C = [BLUE, ORANGE, PURPLE]
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b4", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2.0,
})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def write_csv(out, name, header, rows):
    path = os.path.join(out, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {path}")


def rolling(x, n=25):
    if len(x) < n:
        return np.asarray(x, float)
    k = np.ones(n)
    return np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")


# ------------------------------------------------------------------------------ 1. sweep
def sweep(logs, out, step_dt=0.01):
    pat = re.compile(r"generation_log_(\d+)_s([0-9.]+)_eslr([0-9.]+)\.csv$")
    runs = []
    for f in sorted(glob.glob(os.path.join(logs, "generation_log_*_s*_eslr*.csv"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        g = np.array([float(r["generation"]) for r in rows])
        surv = np.array([float(r["steps_survived_mean"]) for r in rows]) * step_dt
        fit = np.array([float(r["fitness_mean"]) for r in rows])
        runs.append(dict(job=m.group(1), sigma=float(m.group(2)), lr=float(m.group(3)), g=g, surv=surv, fit=fit))
    if not runs:
        print(f"[sweep] no logs/generation_log_<job>_s<sigma>_eslr<lr>.csv found in {logs}, skipped")
        return
    lr_col = {0.01: BLUE, 0.02: ORANGE, 0.03: PURPLE}
    rows = []
    for key, ylabel, fname, ylim in [("surv", "time survived [s] (of 20 s)", "fig1a_sweep_survival.png", (0, 20.5)),
                                     ("fit", "fitness (population mean)", "fig1b_sweep_fitness.png", None)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in sorted(runs, key=lambda r: (r["sigma"], r["lr"]) == (0.05, 0.01)):   # best drawn last
            best = (r["sigma"], r["lr"]) == (0.05, 0.01)
            ax.plot(r["g"], rolling(r[key]), color=lr_col.get(r["lr"], GRAY),
                    ls="-" if r["sigma"] == 0.05 else "--", lw=3.2 if best else 1.6, alpha=1 if best else 0.8,
                    label=f"σ {r['sigma']:g}, lr {r['lr']:g}" + ("  (best)" if best else ""))
        ax.set_xlabel("generation"); ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        h, l = ax.get_legend_handles_labels()
        order = sorted(range(len(l)), key=lambda i: ("best" not in l[i], l[i]))
        ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", fontsize=11, ncol=1)
        save(fig, out, fname)
    for r in runs:
        sm_s, sm_f = rolling(r["surv"]), rolling(r["fit"])
        for i in range(len(r["g"])):
            rows.append([r["job"], r["sigma"], r["lr"], int(r["g"][i]), round(r["surv"][i], 4), round(sm_s[i], 4),
                         round(r["fit"][i], 4), round(sm_f[i], 4)])
    write_csv(out, "fig1_sweep.csv", ["job", "sigma", "adam_lr", "generation", "survival_s", "survival_s_smooth25",
                                      "fitness_mean", "fitness_mean_smooth25"], rows)
    for r in runs:
        print(f"  sweep  σ {r['sigma']:<5g} lr {r['lr']:<5g} job {r['job']}: {len(r['g'])} gens, "
              f"survival last 25 gens {r['surv'][-25:].mean():.1f} s")


# ----------------------------------------------------------------------------- flights
def load_flights(folder):
    eps = sorted((f for f in glob.glob(os.path.join(folder, "ep*.npz")) if os.path.basename(f)[2:-4].isdigit()),
                 key=lambda f: int(os.path.basename(f)[2:-4]))
    if not eps:
        raise SystemExit(f"no ep*.npz in {folder}")
    F = []
    for f in eps:
        d = np.load(f)
        if "att_err" not in d.files:
            raise SystemExit(f"{f} has no attitude data: replay it with jobs/slide_replay.sh")
        k = int(os.path.basename(f)[2:-4])
        T = d["alive"].shape[0]
        for e in range(d["alive"].shape[1]):
            n = int(d["alive"][:, e].sum())
            F.append(dict(ep=k, env=e, n=n, T=T, full=n >= T, dt=float(d["step_dt"]),
                          load=d["load"][:n, e], goal=d["goal"][:n, e], drones=d["drones"][:n, e],
                          att=np.degrees(d["att_err"][:n, e]), load_R=d["load_R"][:n, e], goal_R=d["goal_R"][:n, e],
                          angvel=d["load_angvel"][:n, e]))
    return F


def pick_example(F, ex):
    if ex:
        s = [f for f in F if f["ep"] == ex[0] and f["env"] == ex[1]]
        if not s:
            raise SystemExit(f"no flight ep {ex[0]} env {ex[1]}")
        return s[0]
    full = [f for f in F if f["full"]]
    end = np.array([np.linalg.norm(f["load"][-1] - f["goal"][-1]) for f in full])
    return full[int(np.argsort(end)[len(end) // 2])]


def band(F, key_fn, T):
    """mean, q25, q75, n over the flights still flying at each step."""
    M = np.full((len(F), T), np.nan)
    for i, f in enumerate(F):
        M[i, :f["n"]] = key_fn(f)
    with np.errstate(all="ignore"):
        return np.nanmean(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), np.sum(~np.isnan(M), 0)


# ---------------------------------------------------------------------------- 2. errors
def errors(F, ex, out):
    T, dt = F[0]["T"], F[0]["dt"]
    t = np.arange(T) * dt
    pos = band(F, lambda f: np.linalg.norm(f["load"] - f["goal"], axis=1), T)
    att = band(F, lambda f: f["att"], T)
    ex_pos = np.full(T, np.nan); ex_pos[:ex["n"]] = np.linalg.norm(ex["load"] - ex["goal"], axis=1)
    ex_att = np.full(T, np.nan); ex_att[:ex["n"]] = ex["att"]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)
    for a, (m, q1, q3, _), exv, lab in [(ax[0], pos, ex_pos, "position error [m]"), (ax[1], att, ex_att, "attitude error [°]")]:
        a.fill_between(t, q1, q3, color=BLUE, alpha=0.22, lw=0, label="25–75 % of flights")
        a.plot(t, m, color=BLUE, lw=2.4, label="mean")
        a.plot(t, exv, color=ORANGE, lw=1.2, label="example flight")
        a.set_ylabel(lab); a.set_ylim(bottom=0)
    ax[0].axhline(0.3, color=INK, ls=":", lw=1.2)
    ax[1].axhline(15, color=INK, ls=":", lw=1.2)
    ax[0].legend(loc="upper right", fontsize=11)
    ax[1].set_xlabel("time [s]"); ax[1].set_xlim(0, t[-1])
    fig.align_ylabels(ax)
    save(fig, out, "fig2_errors.png")
    write_csv(out, "fig2_errors.csv",
              ["t_s", "pos_err_mean_m", "pos_err_q25_m", "pos_err_q75_m", "pos_err_example_m",
               "att_err_mean_deg", "att_err_q25_deg", "att_err_q75_deg", "att_err_example_deg", "n_flights_flying"],
              [[round(t[i], 3), *(round(float(x), 5) for x in (pos[0][i], pos[1][i], pos[2][i], ex_pos[i],
                                                                att[0][i], att[1][i], att[2][i], ex_att[i])), int(pos[3][i])]
               for i in range(0, T, 2)])


# --------------------------------------------------------------------------- 3. numbers
def numbers(F, out):
    full = [f for f in F if f["full"]]
    pe = np.array([np.linalg.norm(f["load"][-1] - f["goal"][-1]) for f in full])
    ae = np.array([f["att"][-1] for f in full])
    ax = np.array([np.abs(f["load"][-1] - f["goal"][-1]) for f in full])
    rows = [
        ["flights", len(F), ""],
        ["flights without crash (full 20 s)", f"{100 * len(full) / len(F):.1f}", "%"],
        ["final position error, mean", f"{pe.mean():.3f}", "m"],
        ["final position error, median", f"{np.median(pe):.3f}", "m"],
        ["final attitude error, mean", f"{ae.mean():.1f}", "deg"],
        ["final attitude error, median", f"{np.median(ae):.1f}", "deg"],
        ["ending within 0.3 m", f"{100 * np.mean(pe < 0.3):.1f}", "%"],
        ["ending within 15 deg", f"{100 * np.mean(ae < 15):.1f}", "%"],
        ["ending within 0.3 m AND 15 deg", f"{100 * np.mean((pe < 0.3) & (ae < 15)):.1f}", "%"],
    ]
    for i, a in enumerate("xyz"):
        rows += [[f"final |error| {a}, mean", f"{ax[:, i].mean():.3f}", "m"],
                 [f"final |error| {a}, median", f"{np.median(ax[:, i]):.3f}", "m"]]
    write_csv(out, "fig3_numbers.csv", ["quantity", "value", "unit"], rows)
    print("\n  numbers (final = end of the flights that lasted the full 20 s)")
    for q, v, u in rows:
        print(f"    {q:36s} {v:>8} {u}")
    print()


# --------------------------------------------------------------------------- 4. weights
def weights(folder, ex, out):
    p = os.path.join(folder, f"weights_ep{ex['ep']}.npz")
    if not os.path.exists(p):
        print(f"[weights] {p} missing (replay without --record_weights?), skipped")
        return
    d = np.load(p)
    if "dev" not in d.files:
        print(f"[weights] {p} is from an older replay (no per-swarm drift), skipped")
        return
    t = d["dev_t"] * float(d["step_dt"])
    e = ex["env"]
    n = int(np.searchsorted(t, ex["n"] * ex["dt"], side="right"))
    w0 = float(d["w0_abs"][e])
    dev = 100 * d["dev"][:n, :, e] / w0                  # % of the mean |start weight|
    pair = 100 * d["pair"][:n, :, e] / w0
    t = t[:n]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(dev.shape[1]):
        ax.plot(t, dev[:, i], color=DRONE_C[i], label=f"drone {i + 1}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean |w − w₀| [% of mean |w₀|]")
    ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    save(fig, out, "fig4a_weights_drift.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (i, j) in enumerate([(1, 2), (1, 3), (2, 3)]):
        ax.plot(t, pair[:, k], color=DRONE_C[k], label=f"drone {i} vs {j}")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean |wᵢ − wⱼ| [% of mean |w₀|]")
    ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    save(fig, out, "fig4b_weights_between.png")
    write_csv(out, "fig4_weights.csv",
              ["t_s", "drift_drone1_pct", "drift_drone2_pct", "drift_drone3_pct",
               "diff_d1_d2_pct", "diff_d1_d3_pct", "diff_d2_d3_pct", "mean_abs_w0"],
              [[round(t[i], 3), *(round(float(x), 4) for x in (*dev[i], *pair[i])), round(w0, 5)] for i in range(len(t))])
    print(f"  weights, example flight: drift at the end {'/'.join(f'{x:.0f}' for x in dev[-1])} % (drone 1/2/3), "
          f"drones differ by {'/'.join(f'{x:.0f}' for x in pair[-1])} % (1-2/1-3/2-3) of the mean start weight")
    alld = 100 * d["dev"][-1] / d["w0_abs"][None, :]
    print(f"  weights, all swarms of episode {ex['ep']}: drift at the end, median {np.median(alld):.0f} % "
          f"(range {alld.min():.0f}-{alld.max():.0f} %)")


# ----------------------------------------------------------------------------- 6. swing
def swing_angle(f):
    v = f["load"] - f["drones"].mean(1)
    return np.degrees(np.arccos(np.clip(-v[:, 2] / np.linalg.norm(v, axis=1), -1, 1)))


def swing(F, ex, out):
    dt = ex["dt"]
    t = np.arange(ex["n"]) * dt
    w = np.degrees(np.linalg.norm(ex["angvel"], axis=1))
    sw = swing_angle(ex)
    fig, ax = plt.subplots(3, 1, figsize=(6, 7.5), sharex=True)
    ax[0].plot(t, ex["att"], color=BLUE); ax[0].set_ylabel("attitude\nerror [°]")
    ax[1].plot(t, w, color=ORANGE); ax[1].set_ylabel("payload ang.\nspeed [°/s]")
    ax[2].plot(t, sw, color=PURPLE); ax[2].set_ylabel("swing\nangle [°]")
    ax[2].set_xlabel("time [s]"); ax[2].set_xlim(0, t[-1])
    for a in ax:
        a.set_ylim(bottom=0)
    fig.align_ylabels(ax)
    save(fig, out, "fig6_swing.png")
    write_csv(out, "fig6_swing.csv", ["t_s", "att_err_deg", "payload_ang_speed_deg_s", "swing_angle_deg"],
              [[round(t[i], 3), round(float(ex["att"][i]), 4), round(float(w[i]), 4), round(float(sw[i]), 4)]
               for i in range(len(t))])

    def lagcorr(a, b, max_lag):
        a, b = a - a.mean(), b - b.mean()
        best = (0, 0.0)
        for L in range(-max_lag, max_lag + 1):
            x, y = (a[L:], b[:len(b) - L]) if L >= 0 else (a[:L], b[-L:])
            c = float(np.corrcoef(x, y)[0, 1]) if len(x) > 10 and x.std() > 0 and y.std() > 0 else 0.0
            if abs(c) > abs(best[1]):
                best = (L, c)
        return best

    first = slice(0, int(6 / dt))       # the first 6 s, where the bump is
    L, c = lagcorr(ex["att"][first], w[first], int(1 / dt))
    L2, c2 = lagcorr(ex["att"][first], sw[first], int(1 / dt))
    pk = int(np.argmax(ex["att"][int(1.5 / dt):int(6 / dt)])) + int(1.5 / dt)
    print(f"  swing, example flight (first 6 s): attitude-error peak at {pk * dt:.1f} s; "
          f"corr(att err, ang speed) {c:+.2f} at lag {L * dt:+.2f} s; corr(att err, swing angle) {c2:+.2f} at lag {L2 * dt:+.2f} s")
    cs = []
    for f in F:
        if f["full"]:
            ww = np.degrees(np.linalg.norm(f["angvel"], axis=1))
            cs.append(lagcorr(f["att"][first], ww[first], int(1 / dt))[1])
    print(f"  swing, all full flights: median best-lag corr(att err, ang speed) in the first 6 s = {np.median(cs):+.2f}")


# ----------------------------------------------------------------------------- 5. video
def video(ex, out, every=4, elev=22, azim=-60):
    from matplotlib import animation
    try:   # optional: ffmpeg from pip (pip install --user imageio-ffmpeg) when the system has none
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    dt = ex["dt"]
    idx = np.arange(0, ex["n"], every)
    L, G, Dr, Rl, Rg = ex["load"], ex["goal"][0], ex["drones"], ex["load_R"], ex["goal_R"][0]
    pos_err = np.linalg.norm(L - G, axis=1)
    box = np.array([[x, y, 0] for x, y in [(.275, .225), (.275, -.225), (-.275, -.225), (-.275, .225), (.275, .225)]])
    AX = 0.35                                                    # axis arrow length [m]
    axc = [BLUE, ORANGE, PURPLE]                                 # x, y, z (no red/green)

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    a3 = fig.add_subplot(gs[:, 0], projection="3d")
    pts = np.concatenate([L, Dr.reshape(-1, 3), G[None]])
    c, r = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 0.15
    a3.set_xlim(c[0] - r, c[0] + r); a3.set_ylim(c[1] - r, c[1] + r); a3.set_zlim(max(0, c[2] - r), c[2] + r)
    a3.set_box_aspect((1, 1, 1))
    a3.view_init(elev=elev, azim=azim)          # fixed camera
    a3.set_xlabel("x [m]", labelpad=8); a3.set_ylabel("y [m]", labelpad=8); a3.set_zlabel("z [m]", labelpad=8)
    for i in range(3):   # goal frame: fixed, dashed
        e = Rg[:, i] * AX
        a3.plot([G[0], G[0] + e[0]], [G[1], G[1] + e[1]], [G[2], G[2] + e[2]], "--", color=axc[i], lw=2.5)
    a3.scatter(*G, color=INK, s=40, marker="o")
    trail, = a3.plot([], [], [], color=GRAY, lw=1.2)
    boxl, = a3.plot([], [], [], color=INK, lw=1.6)
    cab = [a3.plot([], [], [], color=INK, lw=0.8)[0] for _ in range(3)]
    dr = [a3.plot([], [], [], "o", color=DRONE_C[i], ms=8)[0] for i in range(3)]
    pax = [a3.plot([], [], [], color=axc[i], lw=3)[0] for i in range(3)]
    from matplotlib.lines import Line2D
    a3.legend([Line2D([], [], color=axc[0], lw=3), Line2D([], [], color=axc[1], lw=3), Line2D([], [], color=axc[2], lw=3),
               Line2D([], [], color=INK, ls="--", lw=2)], ["payload x", "payload y", "payload z", "goal frame (dashed)"],
              loc="upper left", fontsize=11)
    b1 = fig.add_subplot(gs[0, 1]); b2 = fig.add_subplot(gs[1, 1], sharex=b1)
    tt = np.arange(ex["n"]) * dt
    b1.plot(tt, pos_err, color=BLUE); b1.set_ylabel("position\nerror [m]"); b1.set_ylim(bottom=0)
    b2.plot(tt, ex["att"], color=ORANGE); b2.set_ylabel("attitude\nerror [°]"); b2.set_ylim(bottom=0)
    b2.set_xlabel("time [s]"); b1.set_xlim(0, tt[-1])
    c1, c2 = b1.axvline(0, color=INK, lw=1), b2.axvline(0, color=INK, lw=1)
    title = fig.suptitle("t = 0.0 s", x=0.03, ha="left", fontsize=15)

    def upd(k):
        R, p = Rl[k], L[k]
        bw = p + box @ R.T
        boxl.set_data_3d(bw[:, 0], bw[:, 1], bw[:, 2])
        trail.set_data_3d(L[:k + 1, 0], L[:k + 1, 1], L[:k + 1, 2])
        for i in range(3):
            e = R[:, i] * AX
            pax[i].set_data_3d([p[0], p[0] + e[0]], [p[1], p[1] + e[1]], [p[2], p[2] + e[2]])
            dr[i].set_data_3d([Dr[k, i, 0]], [Dr[k, i, 1]], [Dr[k, i, 2]])
            cab[i].set_data_3d([Dr[k, i, 0], p[0]], [Dr[k, i, 1], p[1]], [Dr[k, i, 2], p[2]])
        for cc in (c1, c2):
            cc.set_xdata([k * dt, k * dt])
        title.set_text(f"Example flight (episode {ex['ep']}, swarm {ex['env']}):  t = {k * dt:4.1f} s,  "
                       f"error {pos_err[k]:.2f} m, {ex['att'][k]:.0f}°")
        return []

    anim = animation.FuncAnimation(fig, upd, frames=idx, interval=1000 * dt * every)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fps = round(1 / (dt * every))
    base = os.path.join(out, "fig5_example_flight")
    try:
        anim.save(base + ".mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=4000), dpi=110)
        print(f"saved {base}.mp4")
    except Exception as e:
        print(f"[video] no ffmpeg ({e.__class__.__name__}): mp4 skipped. "
              "Get one with: pip install --user imageio-ffmpeg   (then rerun)")
    anim.save(base + ".gif", writer=animation.PillowWriter(fps=fps), dpi=70)
    print(f"saved {base}.gif")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default=None, help="replay folder from jobs/slide_replay.sh")
    p.add_argument("--logs", default="logs", help="folder with the sweep 2 generation logs")
    p.add_argument("--out", default="slide_figs")
    p.add_argument("--example", type=int, nargs=2, default=None, metavar=("EP", "ENV"))
    p.add_argument("--video", action="store_true", help="also render the example flight (slow, ~5 min)")
    p.add_argument("--no_sweep", action="store_true")
    p.add_argument("--video_only", action="store_true", help="only render the example-flight video")
    p.add_argument("--elev", type=float, default=22, help="video: camera height angle [deg]")
    p.add_argument("--azim", type=float, default=-60, help="video: camera rotation around z [deg]")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.video_only:
        a.no_sweep, a.video = True, True
    if not a.no_sweep:
        sweep(a.logs, a.out)
    if a.replay:
        F = load_flights(a.replay)
        ex = pick_example(F, a.example)
        print(f"  example flight: episode {ex['ep']}, swarm {ex['env']} "
              f"(final error {np.linalg.norm(ex['load'][-1] - ex['goal'][-1]):.2f} m, {ex['att'][-1]:.0f}°)")
        if not a.video_only:
            errors(F, ex, a.out)
            numbers(F, a.out)
            weights(a.replay, ex, a.out)
            swing(F, ex, a.out)
        if a.video:
            video(ex, a.out, elev=a.elev, azim=a.azim)
        with open(os.path.join(a.out, "example_flight.txt"), "w") as f:
            f.write(f"replay {a.replay}\nexample flight: episode {ex['ep']}, swarm {ex['env']}\n")


if __name__ == "__main__":
    main()