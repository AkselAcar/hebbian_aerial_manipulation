#!/usr/bin/env python3
"""How the Hebbian networks change during one flight  (plain python3 + matplotlib).

Reads weights_ep<k>.npz written by replay_hebbian.py --record_weights: the 3 networks
(one per drone, 45 -> 17 -> 17 -> 5) of one swarm, every few steps of the episode.
Colour code everywhere: change of a weight since the start of the flight,
    red = the weight grew, blue = it shrank, faint = barely changed.
Each layer has its own colour scale (printed in the figure), because the three
layers change by very different amounts; the rate plot shows the real sizes.

    python3 plot_network.py --replay <replay folder> --ep 0 --drone 1
        -> network_ep0_drone1.png   network at 5 moments + the 3 weight matrices at the end
        -> weight_rate_ep0.png      weight change per second over time, per layer and drone,
                                    and the total change relative to the start weights
    python3 plot_network.py --replay <replay folder> --ep 0 --drone 1 --animate
        -> network_anim_ep0_drone1.mp4 (.gif if ffmpeg is missing): network + rate, 20 s
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

CMAP = plt.get_cmap("RdBu_r")            # blue = shrank, red = grew
C_LAYER = ["#2a78d6", "#1baf7a", "#eb6834"]
C_DRONE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, GRID = "#4a4a48", "#d9d8d4"
LAYERS = ["input -> hidden 1", "hidden 1 -> hidden 2", "hidden 2 -> output"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 8})


# ----------------------------------------------------------------------------- data
def load(folder, ep):
    d = np.load(os.path.join(folder, f"weights_ep{ep}.npz"))
    if d["W1"].size == 0:
        raise SystemExit("no frames recorded (the recorded swarm crashed at once?)")
    groups = [str(g) for g in d["input_groups"]]
    sizes = [int(n) for n in d["input_sizes"]]
    return d, groups, sizes


def deltas(d, drone):
    """Weight change since the first frame, per layer: list of (F, n_post, n_pre)."""
    return [d[k][:, drone] - d[k][:1, drone] for k in ("W1", "W2", "Wo")]


# ---------------------------------------------------------------------------- graph
def layout(sizes):
    """Node positions: inputs grouped (gaps between groups), hidden layers, outputs."""
    ys, y = [], 0.0
    for n in sizes:
        ys += [y + i for i in range(n)]
        y += n + 1.2
    ys = 1 - np.array(ys) / (y - 1.2)
    h = np.linspace(0.9, 0.1, 17)
    o = np.linspace(0.7, 0.3, 5)
    return [np.c_[np.zeros(45), ys], np.c_[np.ones(17), h], np.c_[2 * np.ones(17), h], np.c_[3 * np.ones(5), o]]


class NetworkPlot:
    """One network drawn in an axis; update(frame) recolours edges and nodes."""

    def __init__(self, ax, d, drone, groups, sizes, scales, labels=True):
        self.ax, self.d, self.drone, self.scales = ax, d, drone, scales
        self.dW = deltas(d, drone)
        pos = layout(sizes)
        segs, self.layer_of = [], []
        for L, (pre, post) in enumerate([(pos[0], pos[1]), (pos[1], pos[2]), (pos[2], pos[3])]):
            for i in range(len(post)):
                for j in range(len(pre)):
                    segs.append([pre[j], post[i]])
                    self.layer_of.append(L)
        self.layer_of = np.array(self.layer_of)
        final = self.values(-1)
        self.order = np.argsort(np.abs(final))          # biggest final changes drawn on top
        self.lc = LineCollection(np.array(segs)[self.order], linewidths=0.4)
        ax.add_collection(self.lc)
        self.nodes = [ax.scatter(p[:, 0], p[:, 1], s=s, c="white", edgecolors=INK, linewidths=0.6, zorder=3)
                      for p, s in zip(pos, [14, 40, 40, 60])]
        ax.set_xlim(-0.9 if labels else -0.1, 3.75 if labels else 3.1)
        ax.set_ylim(-0.1, 1.03)
        ax.axis("off")
        if labels:
            y0 = 0
            for g, n in zip(groups, sizes):
                ax.text(-0.08, pos[0][y0:y0 + n, 1].mean(), g, ha="right", va="center", fontsize=7, color=INK)
                y0 += n
            for k, name in enumerate(d["output_names"]):
                ax.text(3.1, pos[3][k, 1], str(name), ha="left", va="center", fontsize=7, color=INK)
            for x, name in zip([0, 1, 2, 3], ["inputs (45)", "hidden 1 (17)", "hidden 2 (17)", "outputs (5)"]):
                ax.text(x, -0.05, name, ha="center", va="top", fontsize=7, color=INK)
        self.update(0)

    def values(self, f):
        """Edge values for frame f, scaled per layer to [-1, 1]."""
        v = [self.dW[L][f].reshape(-1) / self.scales[L] for L in range(3)]
        return np.clip(np.concatenate(v), -1, 1)

    def update(self, f):
        v = self.values(f)[self.order]
        a = np.abs(v)
        col = CMAP((v + 1) / 2)
        col[:, 3] = 0.03 + 0.97 * a
        self.lc.set_color(col)
        self.lc.set_linewidth(0.15 + 1.8 * a)
        acts = [self.d["s"][f, self.drone], self.d["x1"][f, self.drone], self.d["x2"][f, self.drone],
                self.d["out"][f, self.drone]]
        for sc, x in zip(self.nodes, acts):
            sc.set_facecolor(plt.get_cmap("Greys")(0.08 + 0.8 * np.clip(np.abs(x), 0, 1)))
        return [self.lc, *self.nodes]


def layer_scales(d, drones, q=99):
    """Colour scale per layer: q-th percentile of |change| at the end, over the drones shown."""
    s = []
    for k in ("W1", "W2", "Wo"):
        ch = np.concatenate([np.abs(d[k][-1, i] - d[k][0, i]).ravel() for i in drones])
        s.append(max(np.percentile(ch, q), 1e-9))
    return s


def scale_text(scales):
    return "colour scale per layer (|change| for full colour): " + ", ".join(
        f"{n} ±{s:.3g}" for n, s in zip(["L1", "L2", "L3"], scales))


# ---------------------------------------------------------------------------- plots
def snapshots(d, groups, sizes, drone, folder, ep):
    t = d["t"] * float(d["step_dt"])
    scales = layer_scales(d, [drone])
    want = [1, 2, 5, 10, t[-1]]
    frames = sorted({int(np.argmin(np.abs(t - w))) for w in want if w <= t[-1] + 1e-9})
    fig = plt.figure(figsize=(4.2 * len(frames), 10))
    gs = fig.add_gridspec(2, len(frames), height_ratios=[1.35, 1])
    for k, f in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        NetworkPlot(ax, d, drone, groups, sizes, scales, labels=(k == 0)).update(f)
        ax.set_title(f"t = {t[f]:.1f} s", fontsize=11)
    # final change as matrices
    dW = deltas(d, drone)
    sub = gs[1, :].subgridspec(1, 3, width_ratios=[45, 17, 17], wspace=0.25)
    for L in range(3):
        ax = fig.add_subplot(sub[0, L])
        im = ax.imshow(dW[L][-1], cmap=CMAP, vmin=-scales[L], vmax=scales[L], aspect="auto")
        ax.set_title(f"{LAYERS[L]}: change at t = {t[-1]:.0f} s", loc="left")
        ax.set_ylabel("to neuron"); ax.set_xlabel("from neuron")
        if L == 0:
            edges = np.cumsum([0] + sizes)
            ax.set_xticks((edges[:-1] + edges[1:]) / 2 - 0.5, groups, rotation=60, ha="right", fontsize=7)
            for e in edges[1:-1]:
                ax.axvline(e - 0.5, color="white", lw=1.5)
        if L == 2:
            ax.set_yticks(range(5), [str(n) for n in d["output_names"]], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle(f"Drone {drone + 1}'s network during the flight (episode {ep}, swarm {int(d['env'])}): "
                 f"red = weight grew, blue = shrank since t = 0.   {scale_text(scales)}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(folder, f"network_ep{ep}_drone{drone + 1}.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


def smooth(x, n=20):
    """Moving average that stays correct at both ends."""
    if len(x) <= n:
        return x
    k = np.ones(n)
    return np.convolve(x, k, mode="same") / np.convolve(np.ones_like(x), k, mode="same")


def rates(d, folder, ep):
    dt = float(d["step_dt"])
    R, Ra = d["rate_env"], d["rate_all"]
    tt = np.arange(len(R)) * dt
    t = d["t"] * dt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    print("weight change, per layer (drone 1 / 2 / 3):")
    for L in range(3):
        a = ax[0, L]
        for i in range(R.shape[2]):
            a.plot(tt, smooth(R[:, L, i]), color=C_DRONE[i], lw=1.3, label=f"drone {i + 1}")
        a.plot(tt, smooth(Ra[:, L]), "--", color=INK, lw=1.2, label="mean of all swarms")
        a.set_yscale("log"); a.set_title(f"{LAYERS[L]}: mean |dw/dt|", loc="left")
        a.set_ylabel("per second"); a.grid(color=GRID, lw=0.6)
        b = ax[1, L]
        W0 = d[["W1", "W2", "Wo"][L]]
        rel = np.abs(W0 - W0[:1]).mean(axis=(2, 3)) / np.abs(W0[:1]).mean(axis=(2, 3))   # (F, drones)
        for i in range(rel.shape[1]):
            b.plot(t, 100 * rel[:, i], color=C_DRONE[i], lw=1.6, label=f"drone {i + 1}")
        b.set_title(f"{LAYERS[L]}: total change mean|w(t) - w(0)| / mean|w(0)|", loc="left")
        b.set_ylabel("%"); b.set_xlabel("time [s]"); b.grid(color=GRID, lw=0.6)
        early = R[: int(2 / dt), L].mean(0)
        late = R[-int(10 / dt):, L].mean(0)
        print(f"  {LAYERS[L]:22s} rate first 2 s {'/'.join(f'{x:.3g}' for x in early)} /s, "
              f"last 10 s {'/'.join(f'{x:.3g}' for x in late)} /s, "
              f"total change at the end {'/'.join(f'{x:.0f}' for x in 100 * rel[-1])} %")
    ax[0, 0].legend(loc="lower right"); ax[1, 0].legend(loc="upper left")
    fig.suptitle(f"How fast the weights change (episode {ep}, swarm {int(d['env'])}, Hebbian lr {float(d['lr']):g})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(folder, f"weight_rate_ep{ep}.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


def animate(d, groups, sizes, drone, folder, ep, fps):
    from matplotlib import animation
    dt = float(d["step_dt"])
    t = d["t"] * dt
    scales = layer_scales(d, [drone])
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1])
    ax = fig.add_subplot(gs[0])
    net = NetworkPlot(ax, d, drone, groups, sizes, scales)
    title = ax.set_title(f"Drone {drone + 1}'s network, episode {ep}: t = 00.0 s", fontsize=12, loc="left", pad=8)
    b = fig.add_subplot(gs[1])
    R = d["rate_env"][:, :, drone]
    tt = np.arange(len(R)) * dt
    for L in range(3):
        b.plot(tt, smooth(R[:, L]), color=C_LAYER[L], lw=1.4, label=LAYERS[L])
    b.set_yscale("log"); b.set_xlabel("time [s]"); b.set_ylabel("mean |dw/dt| [1/s]")
    b.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3); b.grid(color=GRID, lw=0.6)
    cur = b.axvline(0, color=INK, lw=1)
    fig.text(0.01, 0.005, "red = weight grew, blue = shrank since t = 0 (faint = unchanged); "
             "node shade = |activation|.  " + scale_text(scales), fontsize=8, color=INK)

    def upd(f):
        arts = net.update(f)
        title.set_text(f"Drone {drone + 1}'s network, episode {ep}: t = {t[f]:4.1f} s")
        cur.set_xdata([t[f], t[f]])
        return [*arts, title, cur]

    anim = animation.FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps, blit=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(folder, f"network_anim_ep{ep}_drone{drone + 1}.mp4")
    try:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), dpi=100)
    except Exception as ex:
        print(f"ffmpeg not usable ({ex.__class__.__name__}), writing a gif instead")
        out = out.replace(".mp4", ".gif")
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=70)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True, help="replay folder with weights_ep<k>.npz")
    p.add_argument("--ep", type=int, default=0)
    p.add_argument("--drone", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--animate", action="store_true")
    p.add_argument("--fps", type=int, default=0, help="animation fps (default: real time)")
    args = p.parse_args()
    d, groups, sizes = load(args.replay, args.ep)
    drone = args.drone - 1
    if args.animate:
        fps = args.fps or max(1, round(1 / (float(d["step_dt"]) * int(d["record_every"]))))
        animate(d, groups, sizes, drone, args.replay, args.ep, fps)
    else:
        snapshots(d, groups, sizes, drone, args.replay, args.ep)
        rates(d, args.replay, args.ep)


if __name__ == "__main__":
    main()