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
  weights_ep<k>.npz                with --record_weights: weights + activations of one swarm
                                   over time, and the weight change rate of all swarms
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
parser.add_argument("--payload_mass", type=float, default=None,
                    help="payload mass in kg for the whole replay (default: env default 1.4 kg)")
parser.add_argument("--record_weights", action="store_true",
                    help="save the 3 networks of one swarm during each episode (weights_ep<k>.npz, for plot_network.py)")
parser.add_argument("--record_env", type=int, default=0, help="with --record_weights: which swarm to record")
parser.add_argument("--record_every", type=int, default=5, help="with --record_weights: keep every k-th step")
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
from isaaclab.utils.math import matrix_from_quat, quat_error_magnitude

import MARL_mav_carry_ext.tasks  # noqa: F401
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg

CRASH_REASONS = ["falcon_fly_low", "payload_fly_low", "illegal_contact", "angle_limit_drone",
                 "angle_limit_load", "cable_collision", "drone_collision", "body_pos_outside"]


INPUT_GROUPS = [  # the 45 inputs of each drone's network (partial observation, history 1)
    ("payload pos", 3), ("payload rot", 9), ("drone id", 3), ("own pos", 3), ("own rot", 9),
    ("own vel", 3), ("own ang vel", 3), ("goal pos err", 3), ("goal rot err", 9)]
OUTPUT_NAMES = ["acc x", "acc y", "acc z", "roll rate", "pitch rate"]


class WeightRecorder:
    """Records the networks of one swarm during an episode (weights before each update,
    neuron activations) and, at every step, the weight change of all swarms."""

    def __init__(self, brains, env, every):
        self.b, self.env, self.every = brains, env, every
        self.frames = {k: [] for k in ["t", "W1", "W2", "Wo", "s", "x1", "x2", "out"]}
        self.rate_env, self.rate_all = [], []
        # all swarms: distance of each drone's weights from the start weights, and between drones
        self.W0 = torch.cat([W.reshape(W.shape[0], -1) for W in (brains.W1, brains.W2, brains.Wo)], 1).clone()
        self.dev_t, self.dev, self.pair = [], [], []

    def ids(self):
        return [a * self.b.n_envs + self.env for a in range(self.b.n_agents)]

    def before(self, t, obs_all, alive_env):
        """Call before brains.act: snapshot of the weights and the activations they produce."""
        self.prev = [W.clone() for W in (self.b.W1, self.b.W2, self.b.Wo)]
        if t % self.every:
            return
        Wf = torch.cat([W.reshape(W.shape[0], -1) for W in self.prev], 1)          # (n_brains, 1139)
        na, ne = self.b.n_agents, self.b.n_envs
        self.dev.append((Wf - self.W0).abs().mean(1).view(na, ne).cpu().numpy())   # (drones, envs)
        Wd = Wf.view(na, ne, -1)
        pairs = [(0, 1), (0, 2), (1, 2)] if na == 3 else [(i, j) for i in range(na) for j in range(i + 1, na)]
        self.pair.append(torch.stack([(Wd[i] - Wd[j]).abs().mean(-1) for i, j in pairs]).cpu().numpy())
        self.dev_t.append(t)
        if not alive_env:
            return
        ids = self.ids()
        W1, W2, Wo = (W[ids] for W in self.prev)
        s = obs_all[ids].to(W1.dtype) / self.b.state_div
        x1 = torch.tanh(torch.bmm(W1, s.unsqueeze(-1))).squeeze(-1)
        x2 = torch.tanh(torch.bmm(W2, x1.unsqueeze(-1))).squeeze(-1)
        out = torch.tanh(torch.bmm(Wo, x2.unsqueeze(-1))).squeeze(-1)
        for k, v in zip(["W1", "W2", "Wo", "s", "x1", "x2", "out"], [W1, W2, Wo, s, x1, x2, out]):
            self.frames[k].append(v.float().cpu().numpy())
        self.frames["t"].append(t)

    def after(self, mask):
        """Call after brains.act: mean |dW| of this step per layer."""
        d = [(W - P).abs().mean(dim=(1, 2)) for W, P in zip((self.b.W1, self.b.W2, self.b.Wo), self.prev)]
        d = torch.stack(d, 1)                                  # (n_brains, 3 layers)
        self.rate_env.append(d[self.ids()].T.cpu().numpy())    # (3 layers, 3 drones)
        m = mask.float().unsqueeze(1)
        self.rate_all.append(((d * m).sum(0) / m.sum().clamp(min=1)).cpu().numpy())

    def save(self, path, step_dt, first_crash_env):
        f = {k: np.stack(v) if v else np.zeros(0) for k, v in self.frames.items()}
        np.savez_compressed(path, **f, step_dt=step_dt, record_every=self.every, env=self.env,
                            first_crash=first_crash_env,
                            rate_env=np.stack(self.rate_env) / step_dt,   # (T, layer, drone), per second
                            rate_all=np.stack(self.rate_all) / step_dt,   # (T, layer), all swarms alive
                            input_groups=np.array([g for g, _ in INPUT_GROUPS]),
                            input_sizes=np.array([n for _, n in INPUT_GROUPS]),
                            output_names=np.array(OUTPUT_NAMES), lr=self.b.lr,
                            dev_t=np.array(self.dev_t), dev=np.stack(self.dev),        # (F, drones, envs)
                            pair=np.stack(self.pair),                                  # (F, 3 pairs, envs)
                            w0_abs=self.W0.abs().mean(1)[: self.b.n_envs].cpu().numpy())


def payload_masses(u, body="load_link"):
    """Payload mass of every env as the simulator currently sees it (None if not readable)."""
    try:
        idx = u.robot.find_bodies(body)[0][0]
    except Exception:
        return None
    getters = (lambda: u.robot.root_physx_view.get_masses(),
               lambda: u.robot.root_view.get_masses(),
               lambda: u.robot.data.body_mass)
    for get in getters:
        try:
            m = get()
            m = torch.as_tensor(getattr(m, "torch", m))
            return m.reshape(m.shape[0], -1)[:, idx].float().cpu()
        except Exception:
            continue
    return None


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
    if args_cli.payload_mass is not None:
        env_cfg.events.randomize_mass.params["mass_distribution_params"] = (args_cli.payload_mass,) * 2
    if args_cli.video:
        configure_camera(env_cfg, args_cli.cam_eye, args_cli.cam_lookat)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    u = env.unwrapped
    agents = u.cfg.possible_agents
    n_agents = len(agents)
    device = u.device
    fps = max(1, round(1.0 / (u.step_dt * args_cli.frame_every)))
    pm = payload_masses(u)
    mass = float(pm.mean()) if pm is not None else float(args_cli.payload_mass or 1.4)
    print(f"payload mass in the simulator: {mass:.3f} kg"
          + ("" if pm is not None else " (could not read it back, value as requested)"), flush=True)
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
        q_load, q_goal = u.load_orientation, u.pose_command_w[:, 3:7]
        snap["att_err"] = quat_error_magnitude(q_goal, q_load).detach().clone()   # rad, same as the reward
        snap["load_R"] = matrix_from_quat(q_load).detach().clone()                 # (n_envs, 3, 3)
        snap["goal_R"] = matrix_from_quat(q_goal).detach().clone()
        try:   # payload velocity in the world frame, read directly from the simulator
            st = u.robot.data.body_com_state_w
            st = getattr(st, "torch", st)[:, u._payload_idx].reshape(u.num_envs, -1)
            snap["load_linvel"], snap["load_angvel"] = st[:, 7:10].clone(), st[:, 10:13].clone()
        except Exception:
            snap["load_linvel"], snap["load_angvel"] = u.load_vel.clone(), u.load_ang_vel.clone()
        return result

    u._get_dones = get_dones_with_snapshot

    brains = BatchedHebbian(lr=args_cli.lr, device=device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = args_cli.which if args_cli.which == "theta" or args_cli.best_gen == -1 else f"best_gen{args_cli.best_gen}"
    if args_cli.payload_mass is not None:
        label += f"_m{args_cli.payload_mass:g}"
    if args_cli.lr != 0.001:
        label += f"_heblr{args_cli.lr:g}"          # e.g. _heblr0 = plasticity switched off
    out_dir = os.path.join(args_cli.results_dir, f"replay_{label}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    summary = [f"genome: {args_cli.which} from {args_cli.results_dir}, Hebbian lr {args_cli.lr}, payload {mass:.3f} kg",
               f"bounding box: ±{getattr(u.cfg, 'bounding_box_threshold', '?')} m (drone positions, any axis)", ""]
    all_survived = []

    for ep in range(args_cli.n_episodes):
        brains.load(genome[None, :], n_envs, n_agents)
        obs, _ = env.reset()
        alive = torch.ones(n_envs, dtype=torch.bool, device=device)
        first_crash = np.full(n_envs, T)
        reasons = np.zeros((n_envs, len(CRASH_REASONS)), dtype=bool)
        rec = {k: [] for k in ["drones", "load", "goal", "actions", "alive",
                               "att_err", "load_R", "goal_R", "load_linvel", "load_angvel"]}
        wrec = WeightRecorder(brains, args_cli.record_env, args_cli.record_every) if args_cli.record_weights else None
        frames, writer = [], None
        if args_cli.video:
            clip_tmp = os.path.join(out_dir, f"ep{ep}_tmp.mp4")
            writer = open_writer(clip_tmp, fps)

        for t in range(T):
            obs_all = torch.cat([obs[a] for a in agents], dim=0)
            mask = alive.repeat(n_agents)
            if wrec:
                wrec.before(t, obs_all, bool(alive[args_cli.record_env]))
            acts = brains.act(obs_all, update_mask=mask) * mask.unsqueeze(1)
            if wrec:
                wrec.after(mask)
            acts = acts.view(n_agents, n_envs, -1)
            obs, rewards, terminated, truncated, _ = env.step({a: acts[i] for i, a in enumerate(agents)})

            alive_before = alive.clone()
            rec["drones"].append(snap["drones"].cpu().numpy())
            rec["load"].append(snap["load"].cpu().numpy())
            rec["goal"].append(snap["goal"].cpu().numpy())
            rec["actions"].append(acts.permute(1, 0, 2).cpu().numpy())    # (n_envs, 3 drones, 5)
            rec["alive"].append(alive_before.cpu().numpy())
            for k in ["att_err", "load_R", "goal_R", "load_linvel", "load_angvel"]:
                rec[k].append(snap[k].float().cpu().numpy())

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
                            att_err=np.stack(rec["att_err"]), load_R=np.stack(rec["load_R"]),
                            goal_R=np.stack(rec["goal_R"]), load_linvel=np.stack(rec["load_linvel"]),
                            load_angvel=np.stack(rec["load_angvel"]),
                            reason_names=np.array(CRASH_REASONS), step_dt=u.step_dt, payload_mass=mass,
                            hebbian_lr=args_cli.lr,
                            bbox=float(getattr(u.cfg, "bounding_box_threshold", 5.0)))
        if wrec:
            wpath = os.path.join(out_dir, f"weights_ep{ep}.npz")
            wrec.save(wpath, u.step_dt, int(first_crash[args_cli.record_env]))
            print(f"[weights] episode {ep}: {wpath}", flush=True)
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