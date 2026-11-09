#!/usr/bin/env python3
"""Diagnostics of trained controllers from replay folders  (plain python3 + matplotlib).

    python3 diagnose_replay.py --replay <folder> [<folder> ...] --out diag
        -> table of key numbers for every folder (printed + diag/diag_table.csv), e.g. to compare
           a normal replay with --freeze_after 5 and --zero_id replays of the same controller
        -> for the FIRST folder:
           diag_orientation.png / .csv   attitude error split into tilt and yaw over time, and
                                         payload tilt reached vs goal tilt asked
           diag_position.png / .csv      signed final error per axis (bias) vs goal coordinate
           diag_reward_terms.png / .csv  reward per term over time and share of the episode return
                                         (needs a replay made with the current replay_hebbian.py)

Needs replays from replay_hebbian.py with attitude data (jobs/slide_replay.sh or jobs/diag_replays.sh).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, PURPLE, GREEN_BLUE, GRAY, INK, GRID = "#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#8a8a86", "#3a3a38", "#dcdbd7"
TERM_C = [BLUE, ORANGE, PURPLE, GREEN_BLUE, GRAY, "#c9a227"]
# reward weights of the hover env (marl_hover_env_cfg.py) -> maximum per step = weight * step_dt
TERM_W = {"pos_reward": 1.5, "ori_reward": 1.5, "action_smoothness": 0.5, "body_rate_penalty": 0.5,
          "force_penalty": 0.5, "downwash_reward": 0.5}
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "legend.fontsize": 11, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "savefig.dpi": 200, "savefig.bbox": "tight"})


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(folder):
    eps = sorted(f for f in glob.glob(os.path.join(folder, "ep*.npz")) if os.path.basename(f)[2:-4].isdigit())
    if not eps:
        raise SystemExit(f"no ep*.npz in {folder}")
    fl, rew_names = [], None
    for f in eps:
        d = np.load(f)
        if "att_err" not in d.files:
            raise SystemExit(f"{f}: no attitude data, replay with the current replay_hebbian.py")
        T = d["alive"].shape[0]
        has_rew = "rew_terms" in d.files
        if has_rew:
            rew_names = [str(x) for x in d["rew_names"]]
        for e in range(d["alive"].shape[1]):
            n = int(d["alive"][:, e].sum())
            Rl, Rg = d["load_R"][:n, e], d["goal_R"][:n, e]
            fl.append(dict(
                n=n, T=T, full=n >= T, dt=float(d["step_dt"]),
                pos=d["load"][:n, e] - d["goal"][:n, e], goal=d["goal"][0, e], start=d["load"][0, e],
                att=np.degrees(d["att_err"][:n, e]),
                tilt=np.degrees(np.arccos(np.clip(np.sum(Rl[:, :, 2] * Rg[:, :, 2], -1), -1, 1))),
                yaw=np.degrees(np.abs(wrap(yaw_of(Rl) - yaw_of(Rg)))),
                load_tilt=np.degrees(np.arccos(np.clip(Rl[:, 2, 2], -1, 1))),
                goal_tilt=float(np.degrees(np.arccos(np.clip(Rg[0, 2, 2], -1, 1)))),
                rew=d["rew_terms"][:n, e] if has_rew else None))
    return fl, rew_names


def at(f, t_s):
    k = min(int(round(t_s / f["dt"])), f["n"] - 1)
    return k


def table_row(name, fl):
    full = [f for f in fl if f["full"]]
    pe = np.array([np.linalg.norm(f["pos"][-1]) for f in full])
    ae = np.array([f["att"][-1] for f in full])
    tl = np.array([f["tilt"][-1] for f in full])
    yw = np.array([f["yaw"][-1] for f in full])
    rmse_p = np.sqrt(np.mean(np.concatenate([np.sum(f["pos"] ** 2, 1) for f in full])))
    rmse_a = np.sqrt(np.mean(np.concatenate([f["att"] ** 2 for f in full])))
    p5, p10 = [np.median([np.linalg.norm(f["pos"][at(f, s)]) for f in full]) for s in (5, 10)]
    bias = np.mean([f["pos"][-1] for f in full], 0)
    return [name, len(fl), f"{100 * len(full) / len(fl):.0f}", f"{pe.mean():.2f}", f"{np.median(pe):.2f}",
            f"{100 * np.mean(pe < 0.3):.0f}", f"{rmse_p:.2f}", f"{ae.mean():.0f}", f"{np.median(ae):.0f}",
            f"{np.median(tl):.0f}", f"{np.median(yw):.0f}", f"{rmse_a:.0f}",
            f"{p5:.2f}", f"{p10:.2f}", f"{np.median(pe):.2f}", *(f"{b:+.2f}" for b in bias)]


HEAD = ["replay", "flights", "full_%", "pos_end_mean_m", "pos_end_median_m", "within_0.3m_%", "pos_RMSE_m",
        "att_end_mean_deg", "att_end_median_deg", "tilt_end_median_deg", "yaw_end_median_deg", "att_RMSE_deg",
        "pos_median_5s_m", "pos_median_10s_m", "pos_median_20s_m", "bias_x_m", "bias_y_m", "bias_z_m"]


def band(fl, key, T):
    M = np.full((len(fl), T), np.nan)
    for i, f in enumerate(fl):
        M[i, :f["n"]] = f[key]
    with np.errstate(all="ignore"):
        return np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)


def orientation(fl, out):
    T, dt = fl[0]["T"], fl[0]["dt"]
    t = np.arange(T) * dt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    rows = [t]
    for key, c, lab in [("tilt", BLUE, "tilt error (payload z vs goal z)"), ("yaw", ORANGE, "yaw error (heading)")]:
        m, q1, q3 = band(fl, key, T)
        ax[0].fill_between(t, q1, q3, color=c, alpha=0.2, lw=0)
        ax[0].plot(t, m, color=c, lw=2, label=lab)
        rows += [m, q1, q3]
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("error [°] (median, 25–75 %)"); ax[0].set_ylim(0, 180)
    ax[0].legend(loc="upper right")
    full = [f for f in fl if f["full"]]
    gt = np.array([f["goal_tilt"] for f in full]); lt = np.array([f["load_tilt"][-1] for f in full])
    ax[1].scatter(gt, lt, s=22, color=PURPLE)
    lim = max(gt.max(), lt.max(), 1) * 1.05
    ax[1].plot([0, lim], [0, lim], "--", color=INK, lw=1, label="tilt reached = tilt asked")
    ax[1].set_xlabel("goal tilt asked [°]"); ax[1].set_ylabel("payload tilt at the end [°]"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "diag_orientation.png")); plt.close(fig)
    with open(os.path.join(out, "diag_orientation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "tilt_med", "tilt_q25", "tilt_q75", "yaw_med", "yaw_q25", "yaw_q75"])
        for i in range(0, T, 5):
            w.writerow([round(r[i], 4) for r in rows])
    slope = np.polyfit(gt, lt, 1)[0] if len(gt) > 2 else np.nan
    print(f"  orientation: final median tilt error {np.median([f['tilt'][-1] for f in full]):.0f}°, "
          f"yaw error {np.median([f['yaw'][-1] for f in full]):.0f}°; payload tilt reached vs asked: slope {slope:.2f}, "
          f"mean tilt reached {lt.mean():.0f}° for {gt.mean():.0f}° asked")


def position(fl, out):
    full = [f for f in fl if f["full"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    for i, a in enumerate("xyz"):
        g = np.array([f["goal"][i] for f in full]); err = np.array([f["pos"][-1, i] for f in full])
        ax[i].scatter(g, err, s=20, color=[BLUE, ORANGE, PURPLE][i])
        ax[i].axhline(0, color=INK, lw=1)
        k = np.polyfit(g, err, 1)
        xx = np.linspace(g.min(), g.max(), 10)
        ax[i].plot(xx, np.polyval(k, xx), "--", color=INK, lw=1)
        ax[i].set_xlabel(f"goal {a} [m]"); ax[i].set_ylabel(f"final error {a} (payload − goal) [m]")
        ax[i].set_title(f"bias {err.mean():+.2f} m, slope {k[0]:+.2f}", loc="left", fontsize=12)
        rows.append((a, err.mean(), np.median(err), k[0], k[1]))
    fig.tight_layout(); fig.savefig(os.path.join(out, "diag_position.png")); plt.close(fig)
    with open(os.path.join(out, "diag_position.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["axis", "bias_mean_m", "bias_median_m", "slope_err_vs_goal", "intercept_m"])
        w.writerows([[a, round(b, 4), round(m, 4), round(s, 4), round(c, 4)] for a, b, m, s, c in rows])
    print("  position: final signed error per axis (mean / slope vs goal): " +
          ", ".join(f"{a} {b:+.2f} m / {s:+.2f}" for a, b, m, s, c in rows) +
          "   (slope -1 = ignores the goal on that axis, 0 = follows it)")


def rewards(fl, names, out):
    if not names or fl[0]["rew"] is None:
        print("  rewards: no per-term rewards in this replay (made before 2026-11-02), skipped")
        return
    T, dt = fl[0]["T"], fl[0]["dt"]
    t = np.arange(T) * dt
    K = len(names)
    M = np.full((len(fl), T, K), np.nan)
    for i, f in enumerate(fl):
        M[i, :f["n"]] = f["rew"]
    with np.errstate(all="ignore"):
        mean_t = np.nanmean(M, 0) / dt                                   # per second
    full = [f for f in fl if f["full"]]
    totals = np.array([np.nansum(f["rew"], 0) for f in full]).mean(0)  # mean episode return per term
    maxes = np.array([TERM_W.get(nm, np.nan) * dt * T for nm in names])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for k, nm in enumerate(names):
        ax[0].plot(t, mean_t[:, k], color=TERM_C[k % 6], lw=1.8, label=nm)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("reward per second (mean of flights)")
    ax[0].legend(loc="center right", fontsize=10)
    y = np.arange(K)
    ax[1].barh(y, maxes, color=GRID, label="maximum")
    ax[1].barh(y, totals, color=[TERM_C[k % 6] for k in range(K)], label="obtained")
    ax[1].set_yticks(y, names); ax[1].invert_yaxis(); ax[1].set_xlabel("episode return (full 20 s flights)")
    for k in range(K):
        ax[1].text(totals[k], k, f" {100 * totals[k] / maxes[k]:.0f}%", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "diag_reward_terms.png")); plt.close(fig)
    with open(os.path.join(out, "diag_reward_terms.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["term", "mean_return_full_flights", "max_return", "pct_of_max", "share_of_total_pct"])
        for k, nm in enumerate(names):
            w.writerow([nm, round(totals[k], 4), round(maxes[k], 4), round(100 * totals[k] / maxes[k], 1),
                        round(100 * totals[k] / totals.sum(), 1)])
    print("  rewards (full flights): " + ", ".join(f"{nm} {100 * totals[k] / maxes[k]:.0f}% of max"
                                                    for k, nm in enumerate(names)) +
          f"; total {totals.sum():.1f} of {np.nansum(maxes):.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", nargs="+", required=True)
    p.add_argument("--out", default="diag")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    first = None
    for folder in a.replay:
        fl, names = load(folder)
        rows.append(table_row(os.path.basename(os.path.normpath(folder)), fl))
        if first is None:
            first = (fl, names)
    with open(os.path.join(a.out, "diag_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(HEAD); w.writerows(rows)
    print("\n" + " | ".join(HEAD))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print(f"\n  details for {a.replay[0]}:")
    orientation(first[0], a.out)
    position(first[0], a.out)
    rewards(first[0], first[1], a.out)
    print(f"\nsaved in {a.out}/")


if __name__ == "__main__":
    main()