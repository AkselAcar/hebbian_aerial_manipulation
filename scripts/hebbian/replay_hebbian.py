"""
Replay a trained Hebbian controller and record what the swarms do.

Loads a genome from a training results folder (OpenAI-ES centre `theta.npy`,
or the best sample `x_best.npy`), runs a few episodes with it, and saves for
every episode:
  - the positions of the 3 drones, the payload and the goal at every step,
  - the commands of the 3 drones,
  - when the swarm first crashed and which termination condition(s) did it,
  - optionally an .mp4 of the episode (--video, needs --viz newton on GPUs
    without RT cores such as izar's V100, see Fuda's visualization guide).

Positions are read inside the env's own _get_dones() call, i.e. BEFORE Isaac
Lab auto-resets a crashed env, so the last saved step is the crash itself.

Typical use (inside the container, see replay_hebbian.sh):
  # statistics + trajectories, no rendering, many envs in parallel
  train_hebbian ... replay_hebbian.py --headless --results_dir results/3217661_p512_eslr0.01 \
      --num_envs 32 --n_episodes 1
  # one env, several episodes, with video (no --headless!)
  ... replay_hebbian.py --results_dir results/3217661_p512_eslr0.01 \
      --num_envs 1 --n_episodes 6 --video --viz newton

Output: <results_dir>/replay_<which>_<timestamp>/
  ep<k>.npz                        trajectories of episode k (all envs)
  ep<k>_env<e>_<reason>_<steps>.mp4  video of env 0 (with --video)
  summary.txt                      survival and crash reasons per env
Plot them with plot_replay.py.
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a trained Hebbian controller")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--results_dir", type=str, required=True,
                    help="training results folder, e.g. results/3217661_p512_eslr0.01")
parser.add_argument("--which", choices=["theta", "best"], default="theta",
                    help="theta = OpenAI-ES search centre (theta.npy); best = best sample ever (x_best.npy)")
parser.add_argument("--best_gen", type=int, default=-1,
                    help="with --which best: generation row of x_best.npy to replay (default: last). "
                         "x_best holds the best-so-far sample at every generation.")
parser.add_argument("--num_envs", type=int, default=16, help="swarms run in parallel per episode")
parser.add_argument("--n_episodes", type=int, default=1, help="episodes run one after the other")
parser.add_argument("--episode_steps", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.001, help="Hebbian learning rate (same as in training!)")
parser.add_argument("--video", action="store_true", help="record an .mp4 of env 0 per episode")
parser.add_argument("--frame_every", type=int, default=2, help="keep every k-th step in the video")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[9.0, -9.0, 7.0])
parser.add_argument("--cam_lookat", type=float, nargs=3, default=[0.0, 0.0, 1.5])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True  # needed for render_mode="rgb_array", before AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch

# Load utils/batched_hebbian.py by its file path: with rendering on, Isaac Sim puts its own
# "utils" package on sys.path, so "from utils.batched_hebbian import ..." can fail.
import importlib.util

_bh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "batched_hebbian.py")
_spec = importlib.util.spec_from_file_location("batched_hebbian", _bh_path)
_bh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bh)
BatchedHebbian = _bh.BatchedHebbian

import MARL_mav_carry_ext.tasks  # noqa: F401
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg

CRASH_REASONS = ["falcon_fly_low", "payload_fly_low", "illegal_contact", "angle_limit_drone",
                 "angle_limit_load", "cable_collision", "drone_collision", "body_pos_outside"]


def load_genome(results_dir, which, best_gen=-1):
    if which == "theta":
        path = os.path.join(results_dir, "theta.npy")
        g = np.load(path)
    else:
        path = os.path.join(results_dir, "x_best.npy")
        g = np.load(path)
        if g.ndim == 2:          # one row per generation (best so far at that generation)
            print(f"x_best.npy has {g.shape[0]} generations, using row {best_gen}")
            g = g[best_gen]
    print(f"loaded {which} from {path}: {g.size} parameters, mean |value| {np.abs(g).mean():.4f}", flush=True)
    return g.astype(np.float64)


def configure_camera(env_cfg, eye, lookat):
    """Point the Isaac Lab camera and the Newton viewer at env 0."""
    try:
        env_cfg.viewer.eye = tuple(eye)
        env_cfg.viewer.lookat = tuple(lookat)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
    except Exception as e:  # older/newer configs may differ
        print(f"[camera] could not set env_cfg.viewer: {e}")
    try:
        from isaaclab_visualizers.newton import NewtonVisualizerCfg
        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(eye=tuple(eye), lookat=tuple(lookat),
                                                           visible_env_indices=[0])]
    except Exception as e:
        print(f"[camera] could not configure the Newton viewer: {e}")


def open_writer(path, fps):
    try:
        import imageio.v2 as imageio
        return imageio.get_writer(path, fps=fps)
    except Exception as e:
        print(f"[video] imageio not available ({e}); frames will be saved as .npz instead")
        return None


def main():
    genome = load_genome(args_cli.results_dir, args_cli.which, args_cli.best_gen)
    n_envs, T = args_cli.num_envs, args_cli.episode_steps

    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = n_envs
    env_cfg.history_len = 1
    if args_cli.video:
        configure_camera(env_cfg, args_cli.cam_eye, args_cli.cam_lookat)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    u = env.unwrapped
    agents = u.cfg.possible_agents
    n_agents = len(agents)
    device = u.device
    fps = max(1, round(1.0 / (u.step_dt * args_cli.frame_every)))
    print(f"num_envs={n_envs}, step_dt={u.step_dt}, bounding box = ±{getattr(u.cfg, 'bounding_box_threshold', '?')} m",
          flush=True)

    # Snapshot of the state inside _get_dones, i.e. before Isaac Lab resets crashed envs.
    original_get_dones = u._get_dones
    snap = {}

    def get_dones_with_snapshot():
        result = original_get_dones()
        snap["drones"] = u.drone_positions.detach().clone()          # (n_envs, 3, 3), relative to env origin
        snap["load"] = u.load_position.detach().clone()               # (n_envs, 3)
        snap["goal"] = u.pose_command_w[:, :3].detach().clone()       # (n_envs, 3)
        snap["flags"] = torch.stack([getattr(u, n).reshape(-1).bool() for n in CRASH_REASONS], 1)
        return result

    u._get_dones = get_dones_with_snapshot

    brains = BatchedHebbian(lr=args_cli.lr, device=device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = args_cli.which if args_cli.which == "theta" or args_cli.best_gen == -1 else f"best_gen{args_cli.best_gen}"
    out_dir = os.path.join(args_cli.results_dir, f"replay_{label}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    summary = [f"genome: {args_cli.which} from {args_cli.results_dir}, Hebbian lr {args_cli.lr}",
               f"bounding box: ±{getattr(u.cfg, 'bounding_box_threshold', '?')} m (drone positions, any axis)", ""]
    all_survived = []

    for ep in range(args_cli.n_episodes):
        brains.load(genome[None, :], n_envs, n_agents)
        obs, _ = env.reset()
        alive = torch.ones(n_envs, dtype=torch.bool, device=device)
        first_crash = np.full(n_envs, T)
        reasons = np.zeros((n_envs, len(CRASH_REASONS)), dtype=bool)
        rec = {k: [] for k in ["drones", "load", "goal", "actions", "alive"]}
        frames, writer = [], None
        if args_cli.video:
            clip_tmp = os.path.join(out_dir, f"ep{ep}_tmp.mp4")
            writer = open_writer(clip_tmp, fps)

        for t in range(T):
            obs_all = torch.cat([obs[a] for a in agents], dim=0)
            mask = alive.repeat(n_agents)
            acts = brains.act(obs_all, update_mask=mask) * mask.unsqueeze(1)
            acts = acts.view(n_agents, n_envs, -1)
            obs, rewards, terminated, truncated, _ = env.step({a: acts[i] for i, a in enumerate(agents)})

            alive_before = alive.clone()
            rec["drones"].append(snap["drones"].cpu().numpy())
            rec["load"].append(snap["load"].cpu().numpy())
            rec["goal"].append(snap["goal"].cpu().numpy())
            rec["actions"].append(acts.permute(1, 0, 2).cpu().numpy())    # (n_envs, 3 drones, 5)
            rec["alive"].append(alive_before.cpu().numpy())

            crashed = terminated[agents[0]].reshape(-1).bool()
            timed_out = truncated[agents[0]].reshape(-1).bool()
            new_crash = (crashed & alive).cpu().numpy()
            if new_crash.any():
                first_crash[new_crash] = t + 1
                reasons[new_crash] = snap["flags"].cpu().numpy()[new_crash]

            if args_cli.video and bool(alive[0]) and t % args_cli.frame_every == 0:
                frame = env.render()
                if frame is not None:
                    frame = np.asarray(frame)
                    if writer is not None:
                        writer.append_data(frame)
                    else:
                        frames.append(frame)

            alive &= ~(crashed | timed_out)
            if not bool(alive.any()):
                break

        # --- save this episode ---------------------------------------------------
        np.savez_compressed(os.path.join(out_dir, f"ep{ep}.npz"),
                            drones=np.stack(rec["drones"]), load=np.stack(rec["load"]),
                            goal=np.stack(rec["goal"]), actions=np.stack(rec["actions"]),
                            alive=np.stack(rec["alive"]), first_crash=first_crash, reasons=reasons,
                            reason_names=np.array(CRASH_REASONS), step_dt=u.step_dt,
                            bbox=float(getattr(u.cfg, "bounding_box_threshold", 5.0)))
        if args_cli.video:
            r0 = "+".join(n for n, f in zip(CRASH_REASONS, reasons[0]) if f) or "full_episode"
            clip = os.path.join(out_dir, f"ep{ep}_env0_{r0}_{first_crash[0]}steps.mp4")
            if writer is not None:
                writer.close()
                os.replace(clip_tmp, clip)
            elif frames:
                np.savez_compressed(clip.replace(".mp4", "_frames.npz"), frames=np.stack(frames))
            print(f"[video] episode {ep}: {clip}", flush=True)

        all_survived.append(first_crash)
        for e in range(n_envs):
            r = ", ".join(n for n, f in zip(CRASH_REASONS, reasons[e]) if f) or "never crashed"
            summary.append(f"ep {ep} env {e:3d}: survived {first_crash[e]:5d} steps | {r}")
        print(f"episode {ep}: mean survival {first_crash.mean():.0f} steps, "
              f"full episodes {np.mean(first_crash >= T):.0%}, "
              f"body_pos_outside in {reasons[:, CRASH_REASONS.index('body_pos_outside')].sum()} of {n_envs} envs",
              flush=True)

    s = np.concatenate(all_survived)
    summary += ["", f"overall: {s.size} swarms, mean survival {s.mean():.0f} steps, "
                    f"full episodes {np.mean(s >= T):.0%}"]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"saved replay to {out_dir}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()