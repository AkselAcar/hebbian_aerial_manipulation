#!/usr/bin/env python3
"""Do the Hebbian weights stabilise during a flight, and which ones run away? (plain python3 + matplotlib)

Reads the weights_ep0.npz written by replay_hebbian.py --record_weights (jobs/weights_replays.sh).

    python3 plot_weights.py --runs results/<runA> results/<runB> ... [--labels a b ...] [--out weights_analysis]

-> <out>/weights_stability.png      all runs together:
       a. distance of the weights from their evolved start values over the flight (all swarms, all drones)
       b. speed of weight change over the flight (-> 0 means the weights have settled)
       c. one swarm: final |change| of each of its 3 x 1139 weights, sorted (a few big ones = "exploding")
   <out>/weights_explode_<label>.png  one per run, for the recorded swarm:
       a. the 20 weights that changed most, over time (change since t = 0)
       b. where the change sits in the first layer (inputs x hidden neurons, drone 1)
       c. per neuron: how much its incoming weights changed vs how often that neuron is saturated
          (|activation| > 0.95). Weights growing into a saturated neuron no longer change its output.
   and a table in the terminal.
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
GRID = "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "font.size": 11, "axes.titlesize": 12, "legend.frameon": False})
LAYERS = [("W1", "in->h1"), ("W2", "h1->h2"), ("Wo", "h2->out")]


def newest(run):
    c = [f for f in glob.glob(os.path.join(run, "replay_theta_*", "weights_ep0.npz"))]
    if not c:
        raise SystemExit(f"no weights_ep0.npz under {run}/replay_theta_* (run jobs/weights_replays.sh first)")
    return max(c, key=os.path.getmtime)


def synapse_name(layer, post, pre, d):
    if layer == 0:
        edges = np.cumsum(d["input_sizes"])
        g = int(np.searchsorted(edges, pre, side="right"))
        return f"{d['input_groups'][g]}[{pre - (edges[g - 1] if g else 0)}] -> h1.{post}"
    if layer == 1:
        return f"h1.{pre} -> h2.{post}"
    return f"h2.{pre} -> {d['output_names'][post]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default="weights_analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = a.labels or [os.path.basename(os.path.normpath(r)).split("_", 1)[-1] for r in a.runs]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    print("run | drift at 5 / 10 / 20 s (mean |W - W0|) | change 10->20 s | change speed at 20 s vs 1 s | "
          "share of the total change in the top 1% / 10% of weights | biggest single change")
    for k, (run, lab) in enumerate(zip(a.runs, labels)):
        path = newest(run)
        d = np.load(path, allow_pickle=True)
        dt, c = float(d["step_dt"]), COLORS[k % len(COLORS)]
        # a. drift, all swarms and drones
        t_dev = d["dev_t"] * dt
        dev = d["dev"].reshape(len(t_dev), -1)                         # (F, drones x envs)
        med = np.median(dev, 1)
        ax[0].plot(t_dev, med, color=c, lw=2, label=lab)
        ax[0].fill_between(t_dev, np.percentile(dev, 25, 1), np.percentile(dev, 75, 1), color=c, alpha=0.13, lw=0)
        # b. speed of change (all layers, all alive swarms), smoothed over 0.5 s
        rate = d["rate_all"].mean(1)
        n = max(int(0.5 / dt), 1)
        rate_s = np.convolve(rate, np.ones(n) / n, mode="valid")
        ax[1].plot(np.arange(len(rate_s)) * dt, rate_s, color=c, lw=1.8)
        # c. per-weight change of the recorded swarm
        t = d["t"] * dt
        W = np.concatenate([d[n_].reshape(len(t), d[n_].shape[1], -1) for n_, _ in LAYERS], 2)   # (F, drones, 1139)
        dW = W - W[0]
        end = np.abs(dW[-1]).reshape(-1)
        srt = np.sort(end)[::-1]
        ax[2].plot(np.arange(1, len(srt) + 1) / len(srt) * 100, srt, color=c, lw=1.8)
        at = lambda s: med[min(np.searchsorted(t_dev, s), len(med) - 1)]
        r_at = lambda s: rate_s[min(int(s / dt), len(rate_s) - 1)]
        top = lambda q: srt[: max(int(len(srt) * q), 1)].sum() / max(srt.sum(), 1e-12)
        print(f"{lab} | {at(5):.4f} / {at(10):.4f} / {at(20):.4f} | {100 * (at(20) / max(at(10), 1e-12) - 1):+.0f}% | "
              f"{100 * r_at(19.4) / max(r_at(1), 1e-12):.0f}% | {100 * top(0.01):.0f}% / {100 * top(0.10):.0f}% | "
              f"{srt[0]:.3f} (mean start |W| {float(np.mean(d['w0_abs'])):.3f})")
        explode_figure(d, dW, t, lab, os.path.join(a.out, f"weights_explode_{lab}.png"))
    ax[0].set_title("Distance from the start weights, mean |W - W0|\n(median over swarms and drones, band 25-75 %)", loc="left")
    ax[1].set_title("Speed of weight change, mean |dW| per second\n(all swarms alive)", loc="left")
    ax[2].set_title("One swarm at the end: |change| of every weight, sorted\n(steep left part = few weights carry the change)", loc="left")
    ax[0].set_xlabel("time [s]"); ax[1].set_xlabel("time [s]"); ax[2].set_xlabel("weights, sorted by change [%]")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    for x in ax:
        x.grid(color=GRID, lw=0.6)
    ax[0].set_ylim(bottom=0); ax[1].set_ylim(bottom=0)
    ax[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "weights_stability.png"), dpi=170)
    print(f"saved {a.out}/weights_stability.png and one weights_explode_<label>.png per run")


def explode_figure(d, dW, t, lab, out):
    n_dr = dW.shape[1]
    shapes = [d[n_].shape[2:] for n_, _ in LAYERS]                      # (post, pre) per layer
    sizes = [s[0] * s[1] for s in shapes]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.25, 1.2, 0.9]})
    # a. top 20 trajectories
    end = np.abs(dW[-1])                                                # (drones, 1139)
    order = np.dstack(np.unravel_index(np.argsort(end, axis=None)[::-1], end.shape))[0][:20]
    cmap = plt.get_cmap("viridis")
    for r, (dr, j) in enumerate(order):
        layer = int(np.searchsorted(offs, j, side="right") - 1)
        post, pre = divmod(int(j - offs[layer]), shapes[layer][1])
        name = f"drone {dr + 1}: {synapse_name(layer, post, pre, d)}"
        ax[0].plot(t, dW[:, dr, j], color=cmap(r / 20), lw=1.3, label=name if r < 8 else None)
    ax[0].axhline(0, color=GRID, lw=1)
    ax[0].set_title(f"{lab}: the 20 weights that changed most (one swarm)", loc="left")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("change since t = 0"); ax[0].grid(color=GRID, lw=0.6)
    ax[0].legend(fontsize=7.5, loc="best", title="largest 8", title_fontsize=8)
    # b. first layer map, drone 1
    m = np.abs(dW[-1, 0, : sizes[0]]).reshape(shapes[0])
    im = ax[1].imshow(m, aspect="auto", cmap="magma", interpolation="nearest")
    edges = np.cumsum(d["input_sizes"])
    for e in edges[:-1]:
        ax[1].axvline(e - 0.5, color="white", lw=0.6)
    mids = edges - np.asarray(d["input_sizes"]) / 2 - 0.5
    ax[1].set_xticks(mids); ax[1].set_xticklabels([str(g) for g in d["input_groups"]], rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("hidden neuron (layer 1)")
    ax[1].set_title("|change| at the end, inputs -> hidden layer 1 (drone 1)", loc="left")
    fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02)
    # c. incoming change per neuron vs saturation of that neuron
    for li, (key, mk, col) in enumerate([("x1", "o", "#2a78d6"), ("x2", "s", "#eb6834"), ("out", "^", "#1baf7a")]):
        act = np.abs(d[key])                                            # (F, drones, n)
        sat = (act > 0.95).mean(0)                                      # share of the flight saturated
        inc = np.abs(dW[-1, :, offs[li]: offs[li + 1]]).reshape(n_dr, *shapes[li]).mean(2)   # mean over pre
        ax[2].scatter(100 * sat.reshape(-1), inc.reshape(-1), s=22, marker=mk, color=col, alpha=0.7,
                      label=["hidden 1", "hidden 2", "outputs"][li])
    ax[2].set_xlabel("time the neuron is saturated, |activation| > 0.95 [%]")
    ax[2].set_ylabel("mean |change| of its incoming weights")
    ax[2].set_title("Per neuron (3 drones)", loc="left"); ax[2].grid(color=GRID, lw=0.6); ax[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()