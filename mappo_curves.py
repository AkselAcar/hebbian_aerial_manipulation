#!/usr/bin/env python3
"""MAPPO learning curves without opening TensorBoard: reads the event files of logs/skrl/mappo_hebbnet/*
and plots the mean total reward of every run (needs the tensorboard package: run with the container's python).

  apptainer exec -B $PWD:/workspace/project ~/containers/isaaclab_3.0.0-beta2.sif \
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/mappo_curves.py \
    --logdir /workspace/project/logs/skrl/mappo_hebbnet --out /workspace/project/mappo_curves.png
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("--logdir", default="logs/skrl/mappo_hebbnet")
p.add_argument("--out", default="mappo_curves.png")
p.add_argument("--tag", default="Total reward (mean)", help="part of the scalar's name to plot")
a = p.parse_args()

runs = sorted(d for d in glob.glob(os.path.join(a.logdir, "*")) if os.path.isdir(d) and not d.endswith("_smoke"))
fig, ax = plt.subplots(figsize=(9, 5))
for d in runs:
    ev = EventAccumulator(d, size_guidance={"scalars": 0}); ev.Reload()
    tags = [t for t in ev.Tags()["scalars"] if a.tag in t]
    if not tags:
        print(f"{os.path.basename(d)}: no scalar containing '{a.tag}'; available: {ev.Tags()['scalars'][:15]}")
        continue
    s = ev.Scalars(tags[0])
    x = np.array([e.step for e in s]); y = np.array([e.value for e in s])
    k = max(1, len(y) // 50)
    ys = np.convolve(y, np.ones(k) / k, mode="valid")
    name = "_".join(os.path.basename(d).split("_")[3:])
    ax.plot(x[k - 1:], ys, lw=1.8, label=name)
    q = max(1, len(y) // 10)
    print(f"{name:20s} [{tags[0]}]  steps {x[-1]:>8d} | first 10%: {y[:q].mean():8.3f} | "
          f"middle 10%: {y[len(y)//2 - q//2: len(y)//2 + q//2 + 1].mean():8.3f} | last 10%: {y[-q:].mean():8.3f}")
ax.set_xlabel("step"); ax.set_ylabel(a.tag); ax.grid(alpha=0.3); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(a.out, dpi=150)
print(f"saved {a.out}  (still rising at the end -> resume; flat for the last third -> export best_agent.pt)")