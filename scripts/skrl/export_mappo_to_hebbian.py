"""Turn a MAPPO checkpoint (train_mappo_hebbnet.py) into a Hebbian run folder.

The MAPPO policy is the Hebbian network with A = B = C = D = 0, so its weights are the start weights
and its observation scaler is the input z-normalisation (frozen from now on). Written to --out:

    theta.npy           the policy weights as a static genome (W1 row by row, then ..., then Wo),
                        so replay_hebbian.py --results_dir <out> flies the MAPPO policy itself
    fixed_weights.npy   the same weights, read by train_hebbian.py --from_mappo <out>
    zstats.json         mean / std of the scaler, same format as scripts/hebbian/zstats.json
    train_args.json     network shape, input scaling and drone, read by replay_hebbian.py

The policy mean (no sampling) is used: mean = scale * tanh(Wo tanh(W1 clip((obs - mean) / std, -5, 5))).
Pure torch, no Isaac needed:

  python scripts/skrl/export_mappo_to_hebbian.py --checkpoint logs/skrl/mappo_hebbnet/<run>/checkpoints/best_agent.pt
"""

import argparse
import json
import os
import re

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="agent_<step>.pt or best_agent.pt of a train_mappo_hebbnet run")
parser.add_argument("--out", default=None, help="output folder (default: results/mappo_<run>_<checkpoint>)")
parser.add_argument("--agent", default=None, help="which agent's modules to export (default: the first one)")
args = parser.parse_args()


def main():
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    meta = {}
    mpath = os.path.join(run_dir, "hebbnet_args.json")
    if os.path.isfile(mpath):
        with open(mpath) as f:
            meta = json.load(f)
    else:
        print(f"WARNING: {mpath} not found, assuming falcon, output scale [5, 5, 5, 2, 2]")
    scale = meta.get("output_scale", [5.0, 5.0, 5.0, 2.0, 2.0])
    if [float(s) for s in scale] != [5.0, 5.0, 5.0, 2.0, 2.0]:
        raise SystemExit(f"output scale {scale} differs from the Hebbian network's [5, 5, 5, 2, 2]")

    agents = list(ckpt.keys())
    uid = args.agent or agents[0]
    pol = ckpt[uid]["policy"]
    # Linear layers of the policy, in order (net_container.0, .2, ...); there must be no biases
    keys = sorted((k for k in pol if re.fullmatch(r"net_container\.\d+\.weight", k)), key=lambda k: int(k.split(".")[1]))
    extra = [k for k in pol if k not in keys and k != "log_std_parameter"]
    if extra:
        raise SystemExit(f"unexpected policy parameters {extra} (biases?): not the Hebbian network shape")
    Ws = [pol[k].double().numpy() for k in keys]
    for a, b in zip(Ws[:-1], Ws[1:]):
        assert b.shape[1] == a.shape[0], f"layer shapes do not chain: {[w.shape for w in Ws]}"
    sizes = [Ws[0].shape[1]] + [w.shape[0] for w in Ws]
    hidden = sizes[1:-1]
    theta = np.concatenate([w.reshape(-1) for w in Ws])        # (post, pre) row by row, as BatchedHebbian

    # the other agents should hold the very same policy (parameter sharing)
    for other in agents:
        if other != uid:
            d = max(float((ckpt[other]["policy"][k].double() - pol[k].double()).abs().max()) for k in keys)
            if d > 0:
                print(f"WARNING: {other}'s policy differs from {uid}'s (max |diff| {d:.3g}): not shared, exporting {uid}")

    sc = ckpt[uid].get("observation_preprocessor")
    if sc is None:
        raise SystemExit("no observation_preprocessor in the checkpoint (it must be RunningStandardScaler)")
    mean = sc["running_mean"].double().numpy()
    std = np.sqrt(sc["running_variance"].double().numpy()) + 1e-8   # skrl: (x - mean) / (sqrt(var) + 1e-8)
    if std.min() < 1e-3:
        print(f"WARNING: inputs {np.where(std < 1e-3)[0].tolist()} have std < 1e-3 (almost constant)")

    run = os.path.basename(run_dir)
    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = args.out or os.path.join("results", f"mappo_{run}_{tag}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "theta.npy"), theta)
    np.save(os.path.join(out, "fixed_weights.npy"), theta)
    zs = {"source": f"skrl RunningStandardScaler of {os.path.abspath(args.checkpoint)} (agent {uid}, "
                    f"{float(sc['current_count']):.0f} samples), frozen",
          "mean": mean.tolist(), "std": std.tolist()}
    with open(os.path.join(out, "zstats.json"), "w") as f:
        json.dump(zs, f, indent=1)
    log_std = pol.get("log_std_parameter")
    targs = {"source": "export_mappo_to_hebbian.py", "checkpoint": os.path.abspath(args.checkpoint),
             "genome": "w", "hidden": hidden, "obs_norm": "z", "drone": meta.get("drone", "falcon"),
             "lr": 0.001, "hebb_decay": 0.0, "hebb_gate": 0.0, "hebb_gate_att": 0.0,
             "pos_weight": meta.get("pos_weight"), "ori_weight": meta.get("ori_weight"),
             "calm_weight": meta.get("calm_weight"), "D": int(theta.size),
             "policy_std": None if log_std is None else torch.exp(log_std).tolist()}
    with open(os.path.join(out, "train_args.json"), "w") as f:
        json.dump(targs, f, indent=1)
    print(f"network {' -> '.join(map(str, sizes))}: {theta.size} weights, mean |w| {np.abs(theta).mean():.3f}, "
          f"max |w| {np.abs(theta).max():.3f} | drone {targs['drone']} | written to {out}")


if __name__ == "__main__":
    main()