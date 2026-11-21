#!/usr/bin/env python3
"""Does the low-level controller do what the network (or a test signal) asks?  (plain python3 + matplotlib)

    python3 check_tracking.py --replay results/<run>/replay_<...> [--env 0 --drone 0 --seconds 6] --out track

Compares, for every drone at every step while its swarm is alive, the command with what the drone did:
    acceleration x, y, z  (world frame, gravity removed, m/s^2)  vs  the drone's measured acceleration
    roll rate p, pitch rate q (body frame, rad/s)                 vs  the drone's measured body rates
Needs a replay made with replay_hebbian.py of Nov 21 or later (acc_meas / rate_meas in ep*.npz).

Output:
  track/tracking_<env>_<drone>.png   command and measurement over time for one drone
  track/tracking_table.csv           per channel, over all drones and swarms:
        gain   slope of measured vs command (1 = tracked, 0 = ignored)
        lag_s  delay that best aligns the two (cross-correlation)
        corr   correlation at that delay (1 = same shape)
        rms_err, rms_cmd   size of the error vs size of the command
Note on the low level (controllers/geometric.py, ACCBR): the acceleration command sets the thrust AND the
desired attitude; the body-rate command only enters as a rate reference next to that attitude loop, so it
is not expected to be tracked as well as the accelerations.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, GRID = "#2a78d6", "#eb6834", "#8a8a86", "#dcdbd7"
CH = ["acc x [m/s²]", "acc y [m/s²]", "acc z [m/s²]", "roll rate p [rad/s]", "pitch rate q [rad/s]"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False, "savefig.dpi": 180, "savefig.bbox": "tight"})


def load(folder):
    eps = sorted(f for f in glob.glob(os.path.join(folder, "ep*.npz")) if os.path.basename(f)[2:-4].isdigit())
    if not eps:
        raise SystemExit(f"no ep*.npz in {folder}")
    out = []
    for f in eps:
        d = np.load(f)
        if "acc_meas" not in d.files:
            raise SystemExit(f"{f}: no acc_meas, replay again with the replay_hebbian.py of Nov 21")
        cmd = d["actions"]                                            # (T, envs, 3, 5)
        meas = np.concatenate([d["acc_meas"], d["rate_meas"][..., :2]], -1)   # (T, envs, 3, 5)
        alive = d["alive"].astype(bool).copy()                        # (T, envs)
        # the step a swarm crashes is recorded after the env reset it: drop it
        for e in range(alive.shape[1]):
            n = int(alive[:, e].sum())
            alive[max(n - 1, 0):, e] = False
        out.append(dict(cmd=cmd, meas=meas, alive=alive, dt=float(d["step_dt"]),
                        signal=str(d["test_signal"]) if "test_signal" in d.files else "none"))
    return out


def best_lag(c, m, max_lag):
    """Lag (steps, >= 0) at which m follows c best, and the correlation there."""
    best = (0, -2.0)
    for k in range(max_lag + 1):
        a, b = c[: len(c) - k], m[k:]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[1]:
            best = (k, r)
    return best


def table(eps, max_lag):
    rows = []
    for ch in range(5):
        pairs, lags, cors = [], [], []
        for d in eps:
            T, E, A, _ = d["cmd"].shape
            for e in range(E):
                n = int(d["alive"][:, e].sum())
                if n < 20:
                    continue
                for a in range(A):
                    c, m = d["cmd"][:n, e, a, ch], d["meas"][:n, e, a, ch]
                    pairs.append((c, m))
                    k, r = best_lag(c, m, max_lag)
                    if r > -2:
                        lags.append(k); cors.append(r)
        if not pairs:
            continue
        k0 = int(np.median(lags)) if lags else 0          # one delay per channel, then gain on aligned data
        cs = [c[: len(c) - k0] for c, _ in pairs]
        ms = [m[k0:] for _, m in pairs]
        c, m = np.concatenate(cs), np.concatenate(ms)
        gain = float(np.dot(c - c.mean(), m - m.mean()) / max(np.sum((c - c.mean()) ** 2), 1e-12))
        lag_s = float(np.median(lags)) * eps[0]["dt"] if lags else np.nan
        rows.append([CH[ch], round(gain, 3), round(lag_s, 3), round(float(np.median(cors)) if cors else np.nan, 3),
                     round(float(np.sqrt(np.mean((m - c) ** 2))), 3), round(float(np.sqrt(np.mean(c ** 2))), 3)])
    return rows


def plot(d, env, drone, seconds, path):
    n = int(d["alive"][:, env].sum())
    k = min(n, int(seconds / d["dt"]))
    t = np.arange(k) * d["dt"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for ch in range(5):
        ax[ch].plot(t, d["cmd"][:k, env, drone, ch], color=BLUE, lw=1.8, label="command")
        ax[ch].plot(t, d["meas"][:k, env, drone, ch], color=ORANGE, lw=1.2, label="measured")
        ax[ch].set_ylabel(CH[ch], fontsize=10)
    ax[0].legend(loc="upper right", ncol=2)
    ax[-1].set_xlabel("time [s]")
    ax[0].set_title(f"drone {drone + 1}, swarm {env} ({d['signal']}), alive {n * d['dt']:.1f} s", fontsize=12)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True)
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--drone", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6.0, help="length of the plotted window")
    p.add_argument("--max_lag_s", type=float, default=0.3)
    p.add_argument("--out", default="track")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = load(a.replay)
    rows = table(eps, int(round(a.max_lag_s / eps[0]["dt"])))
    head = ["channel", "gain", "lag_s", "corr", "rms_err", "rms_cmd"]
    with open(os.path.join(a.out, "tracking_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(rows)
    print(" | ".join(head))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    plot(eps[0], a.env, a.drone, a.seconds, os.path.join(a.out, f"tracking_{a.env}_{a.drone}.png"))
    print(f"saved in {a.out}/  (gain ~1 and corr ~1: tracked; gain ~0: the command has no effect)")


if __name__ == "__main__":
    main()