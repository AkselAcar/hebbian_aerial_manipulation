"""
Hebbian / evolution-strategy training for the flycrane hover task (version 8).

v8: curricula and reward weights (step 2, all off by default):
    --curr_yaw_gens N    heading change asked grows from --curr_yaw0 (15 deg) to the full task over N gens
    --curr_tilt_gens M   then goal tilt grows from 0 to +-45 deg over M gens
    --curr_height_gens N start/goal height range grows from 1 m +- --curr_z0 (0.1 m) to the full 0.5-1.5 m
    --pos_weight_end W   position reward weight rises 1.5 -> W over --pos_weight_gens gens
    --calm_weight w      smoothness, body rate, force and downwash weights all set to w (env: 0.5)
    --collision_off_gens N  no drone-collision termination for N gens, back on over --collision_ramp_gens
    Every curriculum ends on the normal task. The current values, what the reset really asked for
    (start_pos_err_mean, start_att_err_deg_*), and the best possible fitness with the current
    weights (fitness_max_per_drone) are logged every generation.

v7: --lr_schedule cosine decays the OpenAI-ES learning rate from --es_lr to
    --lr_min_frac * es_lr over --lr_total_gens generations (current value logged as es_lr_now).
    Plasticity brakes: --hebb_decay L (1/s) pulls every weight back towards its evolved start
    value with time constant 1/L; --hebb_gate E (m) scales each drone's Hebbian update by
    min(1, |goal position error| / E). All training options are saved in
    results/<run>/train_args.json, and replay_hebbian.py reads the brakes from there.

v6: --mass_range LO HI draws a new payload mass (kg, uniform) for every env at
    every reset, i.e. every generation (default: fixed 1.4 kg as before).
    --init_from <theta.npy> starts the search from a trained genome instead of
    zeros (warm start). The payload masses of each generation are logged
    (payload_mass_min / _mean / _max) so the randomization can be checked.

v5: the controllers of all swarms run as ONE batched GPU computation
    (utils/batched_hebbian.py) instead of one numpy call per drone per env per
    step. Same maths as Controllers.hebbianNNController (checked to ~1e-15),
    so results stay comparable with earlier runs, but generations are much
    faster, which makes larger populations affordable. Fitness, reps and the
    bookkeeping (alive mask, steps survived, crash reasons, ...) also stay on
    the GPU, so the only CPU<->GPU traffic per step is one "anyone alive?" check.
v4: --optimizer chooses CMA-ES (default) or OpenAI-ES.
v3: diagnostics: weight size before/after each episode (w_abs_start /
    w_abs_end), mean command size as a fraction of the limits (act_acc_frac /
    act_rate_frac), first-crash reasons (crash_<condition> = share of crashed swarms).
v2: alive mask (reward counts only until a swarm's first termination),
    parallel reps (env j runs genome j % pop_size), steps survived logging.

Genome (--genome): w_abcd = [start weights | A | B | C | D] (5 x 1139 = 5695),
                   w      = start weights only, no plasticity (1139).

Fitness scales: per drone, the maximum for a perfect 2000-step episode is
100 (6 reward terms, weights summing to 5 x step_dt 0.01 x 2000 steps). Summed
over the 3 drones the maximum is 300.
"""

import argparse
import csv
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hebbian/ES training for flycrane hover (v5, batched)")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--pop_size", type=int, default=2, help="Population size (must be even for OpenAI-ES)")
parser.add_argument("--n_reps", type=int, default=3, help="Evaluations per genome (run in parallel)")
parser.add_argument("--n_generations", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=50, help="Max steps per episode")
parser.add_argument("--genome", choices=["w_abcd", "w"], default="w_abcd",
                    help="w_abcd: evolve start weights + ABCD | w: static baseline (weights only)")
parser.add_argument("--optimizer", choices=["cmaes", "openai_es"], default="cmaes")
parser.add_argument("--sigma0", type=float, default=0.1,
                    help="CMA-ES: initial step size (adapts). OpenAI-ES: noise std (fixed).")
parser.add_argument("--es_lr", type=float, default=0.01, help="OpenAI-ES only: learning rate")
parser.add_argument("--es_optimizer", choices=["adam", "sgd"], default="adam",
                    help="OpenAI-ES only: adam (Salimans et al.) or sgd = plain gradient step (Najarro & Risi)")
parser.add_argument("--weight_decay", type=float, default=0.005, help="OpenAI-ES only: L2 penalty on the genome")
parser.add_argument("--lr", type=float, default=0.001, help="Hebbian learning rate (paper: 0.1)")
parser.add_argument("--mass_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="payload mass range in kg, new mass every reset (default: fixed 1.4 kg)")
parser.add_argument("--init_from", type=str, default=None,
                    help="start genome, e.g. results/<run>/theta.npy (default: zeros)")
parser.add_argument("--lr_schedule", choices=["const", "cosine"], default="const",
                    help="OpenAI-ES learning rate: constant, or cosine decay to lr_min_frac * es_lr")
parser.add_argument("--lr_total_gens", type=int, default=1000, help="cosine: generations to reach the minimum")
parser.add_argument("--lr_min_frac", type=float, default=0.1, help="cosine: final lr as a fraction of es_lr")
parser.add_argument("--hebb_decay", type=float, default=0.0,
                    help="plasticity brake: weights relax to their start values at this rate [1/s] (0 = off)")
parser.add_argument("--hebb_gate", type=float, default=0.0,
                    help="plasticity brake: Hebbian update scaled by min(1, |goal pos error| / this) [m] (0 = off)")
# --- step 2: curricula and reward weights (all off by default) -------------------------------
parser.add_argument("--curr_yaw_gens", type=int, default=0,
                    help="heading curriculum: max heading change grows from --curr_yaw0 to the full task over N gens")
parser.add_argument("--curr_yaw0", type=float, default=15.0, help="heading curriculum: first max heading change [deg]")
parser.add_argument("--curr_tilt_gens", type=int, default=0,
                    help="tilt curriculum: goal roll/pitch grow from 0 to the full task over N gens, "
                         "starting when the heading curriculum ends")
parser.add_argument("--curr_height_gens", type=int, default=0,
                    help="height curriculum: start and goal height range grows from mid +- --curr_z0 to the full task")
parser.add_argument("--curr_z0", type=float, default=0.1, help="height curriculum: first half-range [m]")
parser.add_argument("--pos_weight_end", type=float, default=None,
                    help="position reward weight rises linearly from the env value (1.5) to this")
parser.add_argument("--pos_weight_gens", type=int, default=500, help="generations for that rise")
parser.add_argument("--calm_weight", type=float, default=None,
                    help="weight of each calm-flight term (smoothness, body rate, force, downwash; env: 0.5)")
parser.add_argument("--collision_off_gens", type=int, default=0,
                    help="no drone-collision termination for the first N gens, then switched back on gradually")
parser.add_argument("--collision_ramp_gens", type=int, default=200,
                    help="generations over which the collision distance goes from 0 back to the env value (0.6 m)")
parser.add_argument("--tag", type=str, default="",
                    help="Optional label added to the log/results names, e.g. eslr0.001_s0.02")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.EA import CMAes, OpenAIES
from utils.batched_hebbian import BatchedHebbian

import MARL_mav_carry_ext.tasks  # noqa: F401
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg

# The 8 termination conditions of the hover env (see _get_dones in
# marl_hover_env.py). Each is a True/False tensor per env, recomputed on every
# env.step(); several can be True at once.
CRASH_REASONS = [
    "falcon_fly_low",     # a drone below 10 cm
    "payload_fly_low",    # payload below 10 cm
    "illegal_contact",    # contact force above threshold
    "angle_limit_drone",  # cable too steep relative to its drone
    "angle_limit_load",   # cable too steep relative to the payload
    "cable_collision",    # cables too close to each other
    "drone_collision",    # drones too close to each other
    "body_pos_outside",   # a drone outside the bounding box
]


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


def flat(x):
    """Env outputs can be (n_envs,) or (n_envs, 1): make them (n_envs,)."""
    return x.reshape(-1)


def ramp(gen, start, n, a, b):
    """a before generation `start`, b from `start + n` on, linear in between."""
    if gen < start:
        return a
    if n <= 0 or gen >= start + n:
        return b
    return a + (b - a) * (gen - start) / n


CALM_TERMS = ("action_smoothness_weight", "body_rate_penalty_weight", "force_penalty_weight", "downwash_rew_weight")


class Curriculum:
    """Step 2: changes the task before each generation's env.reset().

    Every curriculum ends exactly on the env's normal values, so after it the task is the usual one.
    - heading: the start heading (reset_base yaw) and the goal heading (goal_range yaw) are both drawn
      in +-h/2, so the heading change asked is at most h; h grows from curr_yaw0 to the full task.
    - tilt: goal roll/pitch range grows from 0 to the full +-45 deg after the heading curriculum.
    - height: start z and goal z both drawn in mid +- hz; hz grows from curr_z0 to the full half-range.
    - position weight: pos_track_weight rises linearly to pos_weight_end.
    - calm weight: the 4 calm-flight weights are set once to calm_weight.
    - collision: drone_collision_threshold is 0 (never triggers) for collision_off_gens, then ramps back.
    The env reads all of these at every reset / step, so changing them on the fly is enough.
    """

    def __init__(self, u, args):
        self.u, self.a = u, args
        try:  # the lab's own curricula change event ranges this way (mdp_llc/curriculums.py)
            self.reset_term = u.event_manager.get_term_cfg("reset_base")
        except Exception:
            self.reset_term = u.cfg.events.reset_base
        pr, g = self.reset_term.params["pose_range"], u.cfg.goal_range
        self.full = dict(start_yaw=tuple(pr["yaw"]), goal_yaw=tuple(g["yaw"]),
                         start_z=tuple(pr["z"]), goal_z=tuple(g["pos_z"]),
                         roll=tuple(g["roll"]), pitch=tuple(g["pitch"]),
                         pos_w=float(u.cfg.pos_track_weight), coll=float(u.cfg.drone_collision_threshold))
        if args.calm_weight is not None:
            for k in CALM_TERMS:
                setattr(u.cfg, k, args.calm_weight)
        self.active = any([args.curr_yaw_gens, args.curr_tilt_gens, args.curr_height_gens,
                           args.pos_weight_end is not None, args.calm_weight is not None, args.collision_off_gens])

    @staticmethod
    def _sym(full, half):
        """Symmetric range +-half around the centre of `full`, never wider than `full`."""
        mid, fh = 0.5 * (full[0] + full[1]), 0.5 * (full[1] - full[0])
        h = min(half, fh)
        return (mid - h, mid + h)

    def apply(self, gen):
        a, f, u = self.a, self.full, self.u
        pr, g = self.reset_term.params["pose_range"], u.cfg.goal_range
        s = {}
        # heading: h = max heading change; start and goal each get +-h/2
        h_full = 0.5 * (f["start_yaw"][1] - f["start_yaw"][0]) + 0.5 * (f["goal_yaw"][1] - f["goal_yaw"][0])
        h = ramp(gen, 0, a.curr_yaw_gens, np.radians(a.curr_yaw0), h_full) if a.curr_yaw_gens else h_full
        pr["yaw"], g["yaw"] = self._sym(f["start_yaw"], h / 2), self._sym(f["goal_yaw"], h / 2)
        s["curr_heading_change_max_deg"] = np.degrees(min(h, h_full))
        # tilt (after the heading curriculum)
        t_full = 0.5 * (f["roll"][1] - f["roll"][0])
        t = ramp(gen, a.curr_yaw_gens, a.curr_tilt_gens, 0.0, t_full) if a.curr_tilt_gens else t_full
        g["roll"], g["pitch"] = self._sym(f["roll"], t), self._sym(f["pitch"], t)
        s["curr_tilt_max_deg"] = np.degrees(t)
        # height
        z_full = 0.5 * (f["goal_z"][1] - f["goal_z"][0])
        hz = ramp(gen, 0, a.curr_height_gens, a.curr_z0, z_full) if a.curr_height_gens else z_full
        pr["z"], g["pos_z"] = self._sym(f["start_z"], hz), self._sym(f["goal_z"], hz)
        s["curr_z_half_range"] = min(hz, z_full)
        # reward weights
        if a.pos_weight_end is not None:
            u.cfg.pos_track_weight = ramp(gen, 0, a.pos_weight_gens, f["pos_w"], a.pos_weight_end)
        s["pos_weight_now"] = float(u.cfg.pos_track_weight)
        s["calm_weight_now"] = float(getattr(u.cfg, CALM_TERMS[0]))
        # drone collision
        if a.collision_off_gens:
            u.cfg.drone_collision_threshold = ramp(gen, a.collision_off_gens, a.collision_ramp_gens, 0.0, f["coll"])
        s["collision_dist_now"] = float(u.cfg.drone_collision_threshold)
        try:  # some Isaac Lab versions keep a copy: hand the changed term back
            u.event_manager.set_term_cfg("reset_base", self.reset_term)
        except Exception:
            pass
        return s

    def fitness_max(self, T):
        """Best possible fitness per drone with the current weights (all terms at 1 every step)."""
        c = self.u.cfg
        w = c.pos_track_weight + c.ori_track_weight + sum(getattr(c, k) for k in CALM_TERMS)
        return w * self.u.step_dt * T


def start_errors(obs0):
    """What the reset actually asked for (from drone 0's observation): start distance to the goal,
    start attitude error and start payload height. Used to check that the curriculum takes effect."""
    pos_err = obs0[:, 33:36].norm(dim=1)
    R = obs0[:, 36:45].reshape(-1, 3, 3)                      # goal rotation x payload rotation^T
    cos = ((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)
    att = torch.rad2deg(torch.arccos(cos))
    z = obs0[:, 2]
    return {"start_pos_err_mean": float(pos_err.mean()), "start_att_err_deg_mean": float(att.mean()),
            "start_att_err_deg_p90": float(torch.quantile(att.float(), 0.9)),
            "start_payload_z_min": float(z.min()), "start_payload_z_max": float(z.max())}


CURR_FIELDS = ["curr_heading_change_max_deg", "curr_tilt_max_deg", "curr_z_half_range",
               "pos_weight_now", "calm_weight_now", "collision_dist_now", "fitness_max_per_drone",
               "start_pos_err_mean", "start_att_err_deg_mean", "start_att_err_deg_p90",
               "start_payload_z_min", "start_payload_z_max"]


def main():
    n_input, n_hidden, n_output = 45, 17, 5

    pop = args_cli.pop_size
    n_reps = args_cli.n_reps
    n_envs = pop * n_reps  # one environment per (genome, rep) pair
    T = args_cli.episode_steps

    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = n_envs
    env_cfg.history_len = 1
    if args_cli.mass_range is not None:
        # The env already has a mass event (fixed 1.4 kg, applied once at startup).
        # Give it a range and run it at every reset instead.
        mass_ev = env_cfg.events.randomize_mass
        mass_ev.params["mass_distribution_params"] = tuple(args_cli.mass_range)
        mass_ev.mode = "reset"
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    agents = u.cfg.possible_agents
    n_agents = len(agents)
    device = u.device

    print("=== SETUP ===")
    print(f"pop_size={pop}, n_reps={n_reps} -> num_envs={n_envs}, brains={n_envs * n_agents}, device={device}")
    print(f"env max_episode_length={u.max_episode_length}, your episode_steps={T}")
    print("=============", flush=True)

    n_synapses = n_hidden * n_input + n_hidden * n_hidden + n_output * n_hidden  # 1139
    D = 5 * n_synapses if args_cli.genome == "w_abcd" else n_synapses             # 5695 or 1139
    x0 = np.zeros(D)  # outputs start at 0 = hold position
    if args_cli.init_from:
        x0 = np.load(args_cli.init_from).astype(float).reshape(-1)
        if x0.size != D:
            raise ValueError(f"--init_from has {x0.size} values, genome {args_cli.genome} needs {D}")
        print(f"warm start from {args_cli.init_from} (mean |value| {np.abs(x0).mean():.4f})", flush=True)

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    if args_cli.tag:
        run_id = f"{run_id}_{args_cli.tag}"   # e.g. generation_log_3217000_eslr0.001_s0.02.csv
    results_dir = f"./results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    ea_params = {"bounds": (-5, 5), "D": D, "pop_size": pop,
                 "sigma0": args_cli.sigma0, "x0": x0,
                 "es_lr": args_cli.es_lr, "weight_decay": args_cli.weight_decay,
                 "es_optimizer": args_cli.es_optimizer, "lr_schedule": args_cli.lr_schedule,
                 "lr_total_gens": args_cli.lr_total_gens, "lr_min_frac": args_cli.lr_min_frac}
    if args_cli.optimizer == "cmaes":
        learner = CMAes(ea_params, output_dir=results_dir)
    else:
        learner = OpenAIES(ea_params, output_dir=results_dir)
    print(f"optimizer={args_cli.optimizer}, es_optimizer={args_cli.es_optimizer}, "
          f"es_lr={args_cli.es_lr}, weight_decay={args_cli.weight_decay}")
    print(f"genome={args_cli.genome}, D={D}, sigma0={args_cli.sigma0}, "
          f"lr={args_cli.lr}, results in {results_dir}", flush=True)
    print(f"payload mass: {'random ' + str(args_cli.mass_range) + ' kg, new every reset' if args_cli.mass_range else 'fixed (env default 1.4 kg)'}",
          flush=True)

    brains = BatchedHebbian(n_input, n_hidden, n_output, lr=args_cli.lr, device=device,
                            decay=args_cli.hebb_decay * u.step_dt)       # per-second rate -> per step
    print(f"lr schedule {args_cli.lr_schedule} (total {args_cli.lr_total_gens} gens, min {args_cli.lr_min_frac} x es_lr) | "
          f"plasticity brakes: decay {args_cli.hebb_decay}/s, gate {args_cli.hebb_gate} m", flush=True)
    with open(os.path.join(results_dir, "train_args.json"), "w") as f:
        json.dump({**vars(args_cli), "step_dt": u.step_dt, "D": D}, f, indent=1, default=str)
    scale = brains.action_scale                                   # [5, 5, 5, 2, 2]
    curr = Curriculum(u, args_cli)
    if curr.active:
        print(f"curricula: heading {args_cli.curr_yaw_gens} gens (from {args_cli.curr_yaw0} deg), "
              f"tilt {args_cli.curr_tilt_gens} gens, height {args_cli.curr_height_gens} gens (from +-{args_cli.curr_z0} m), "
              f"pos weight -> {args_cli.pos_weight_end} over {args_cli.pos_weight_gens} gens, "
              f"calm weight {args_cli.calm_weight}, collision off {args_cli.collision_off_gens} gens "
              f"(+{args_cli.collision_ramp_gens} ramp)", flush=True)

    log_path = f"./logs/generation_log_{run_id}.csv"
    log_fields = [
        "generation",
        "fitness_mean", "fitness_std", "fitness_best",        # per drone (max 100)
        "fitness_mean_3drones", "fitness_best_3drones",       # summed scale (max 300)
        "steps_survived_mean", "steps_survived_best",         # steps until first crash
        "frac_full_episode",                                  # evaluations that never crashed
        "stuck_at_start",                                     # envs terminating on step 0
        "sat_frac",                                           # share of actions at the limits
        "w_abs_start",                                        # mean |weight| before the episode
        "w_abs_end",                                          # mean |weight| at episode end
        "act_acc_frac", "act_rate_frac",                      # mean |command| / limit
        "steps_simulated",                                    # episode length actually run
        "genome_variance_mean", "cma_sigma", "seconds",       # cma_sigma = sigma of either optimizer
        "es_step_norm",                                       # OpenAI-ES: size of the last update
        "payload_mass_min", "payload_mass_mean", "payload_mass_max",  # kg, this generation
        "es_lr_now",                                          # OpenAI-ES learning rate used this generation
    ] + CURR_FIELDS + [f"crash_{name}" for name in CRASH_REASONS]   # step 2 values, share of crashed swarms
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    for gen in range(args_cli.n_generations):
        gen_start_t = time.time()

        # --- one brain per (drone, env), all built from the population at once --
        # Env j runs genome j % pop: its 3 drones and its reps start from the same
        # evolved weights and rules, but their Hebbian updates are independent.
        brains.load(learner.x_new, n_envs, n_agents)
        w_abs_start = brains.mean_abs_w1(agent=0)

        curr_now = curr.apply(gen)       # step 2: set this generation's task (no-op when all off)
        curr_now["fitness_max_per_drone"] = curr.fitness_max(T)
        obs, _ = env.reset()
        pm = payload_masses(u)   # masses drawn by this reset
        curr_now.update(start_errors(obs[agents[0]]))
        pm_stats = (float(pm.min()), float(pm.mean()), float(pm.max())) if pm is not None else (np.nan,) * 3
        fitness_env = torch.zeros(n_envs, device=device)               # reward until first termination
        alive = torch.ones(n_envs, dtype=torch.bool, device=device)    # False after first termination
        steps_survived = torch.full((n_envs,), T, dtype=torch.long, device=device)
        stuck_at_start = 0
        steps_simulated = 0
        sat_count = torch.zeros((), device=device)
        act_abs_sum = torch.zeros(n_output, device=device)             # sum of |command| / limit
        act_rows = torch.zeros((), device=device)
        crash_counts = torch.zeros(len(CRASH_REASONS), device=device)
        n_crashed = torch.zeros((), device=device)

        for t in range(T):
            # all drones of all envs in one batch: brain b = agent * n_envs + env
            obs_all = torch.cat([obs[a] for a in agents], dim=0)
            mask = alive.repeat(n_agents)
            gate = None
            if args_cli.hebb_gate > 0:                         # each drone sees its own goal error (obs 33:36, m)
                gate = (obs_all[:, 33:36].norm(dim=1) / args_cli.hebb_gate).clamp(max=1.0)
            acts = brains.act(obs_all, update_mask=mask, gate=gate)   # crashed swarms: weights frozen
            acts = acts * mask.unsqueeze(1)                    # crashed swarms: zero commands

            m = mask.unsqueeze(1).float()
            sat_count += ((acts.abs() > 0.95 * scale).float() * m).sum()
            act_abs_sum += ((acts.abs() / scale) * m).sum(0)
            act_rows += mask.sum()

            acts = acts.view(n_agents, n_envs, n_output)
            actions = {a: acts[i] for i, a in enumerate(agents)}

            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_simulated = t + 1

            r = flat(rewards[agents[0]]).float()
            fitness_env += r * alive              # the step on which a swarm crashes still counts

            crashed = flat(terminated[agents[0]]).bool()
            timed_out = flat(truncated[agents[0]]).bool()
            new_crash = crashed & alive
            steps_survived[new_crash] = t + 1
            n_crashed += new_crash.sum()
            if t == 0:
                stuck_at_start = int(new_crash.sum())

            # Why did these swarms crash? The env's termination flags still hold
            # the values computed during this step, before the auto-reset.
            for k, name in enumerate(CRASH_REASONS):
                flag = getattr(u, name, None)
                if flag is not None:
                    crash_counts[k] += (flat(flag).bool() & new_crash).sum()

            alive &= ~(crashed | timed_out)
            if not bool(alive.any()):
                break  # every env has terminated once: nothing left to measure

        # --- back to numpy for the optimizer and the log ------------------------
        fitness_env = fitness_env.cpu().numpy()
        steps_survived = steps_survived.cpu().numpy()
        n_act = max(float(act_rows), 1.0)
        sat_frac = float(sat_count) / (n_act * n_output)
        act_frac = (act_abs_sum / n_act).cpu().numpy()
        act_acc_frac = float(act_frac[:3].mean())
        act_rate_frac = float(act_frac[3:].mean())
        w_abs_end = brains.mean_abs_w1(agent=0)
        crash_frac = (crash_counts / max(float(n_crashed), 1.0)).cpu().numpy()

        # --- average the reps of each genome ---------------------------------
        # Env j = rep * pop + i, so reshaping to (n_reps, pop) puts rep in the
        # row and genome i in the column.
        fitnesses = fitness_env.reshape(n_reps, pop).mean(axis=0)
        survived = steps_survived.reshape(n_reps, pop).mean(axis=0)

        fitness_mean = float(np.mean(fitnesses))
        fitness_std = float(np.std(fitnesses))
        fitness_best = float(np.max(fitnesses))
        genome_variance_mean = float(np.var(learner.x_new, axis=0).mean())
        sigma = float(learner.sigma)

        learner.f_new = -fitnesses  # both optimizers minimize; we maximize reward
        learner.x_new = learner.get_new_genome()
        learner.save_checkpoint()

        gen_time_s = time.time() - gen_start_t
        frac_full = float(np.mean(steps_survived >= T))

        log_writer.writerow({
            "generation": gen,
            "fitness_mean": fitness_mean,
            "fitness_std": fitness_std,
            "fitness_best": fitness_best,
            "fitness_mean_3drones": n_agents * fitness_mean,
            "fitness_best_3drones": n_agents * fitness_best,
            "steps_survived_mean": float(np.mean(survived)),
            "steps_survived_best": float(np.max(survived)),
            "frac_full_episode": frac_full,
            "stuck_at_start": stuck_at_start,
            "sat_frac": sat_frac,
            "w_abs_start": w_abs_start,
            "w_abs_end": w_abs_end,
            "act_acc_frac": act_acc_frac,
            "act_rate_frac": act_rate_frac,
            "steps_simulated": steps_simulated,
            "genome_variance_mean": genome_variance_mean,
            "cma_sigma": sigma,
            "seconds": gen_time_s,
            "es_step_norm": getattr(learner, "last_step_norm", 0.0),
            "payload_mass_min": pm_stats[0], "payload_mass_mean": pm_stats[1], "payload_mass_max": pm_stats[2],
            "es_lr_now": getattr(learner, "lr", np.nan),       # lr of the update just made
            **curr_now,
            **{f"crash_{name}": float(crash_frac[k]) for k, name in enumerate(CRASH_REASONS)},
        })
        log_file.flush()

        print(f"Gen {gen} done | fitness mean {fitness_mean:.3f} best {fitness_best:.3f} "
              f"(x3: {n_agents * fitness_best:.1f}) | survived mean {np.mean(survived):.0f} "
              f"best {np.max(survived):.0f} steps | full episodes {frac_full:.1%} | "
              f"stuck_at_start {stuck_at_start} | sim steps {steps_simulated} | "
              f"sat {sat_frac:.1%} | |w| {w_abs_start:.3f}->{w_abs_end:.3f} | "
              f"cmd acc {act_acc_frac:.1%} rate {act_rate_frac:.1%} of limit | "
              f"sigma {sigma:.3f} | payload {pm_stats[0]:.2f}-{pm_stats[2]:.2f} kg | {gen_time_s:.1f}s", flush=True)
        if curr.active:
            print(f"    task: heading change <= {curr_now['curr_heading_change_max_deg']:.0f} deg, "
                  f"tilt <= {curr_now['curr_tilt_max_deg']:.0f} deg, z +-{curr_now['curr_z_half_range']:.2f} m, "
                  f"pos w {curr_now['pos_weight_now']:.2f}, calm w {curr_now['calm_weight_now']:.2f}, "
                  f"collision {curr_now['collision_dist_now']:.2f} m | asked at start: "
                  f"{curr_now['start_pos_err_mean']:.2f} m, {curr_now['start_att_err_deg_mean']:.0f} deg "
                  f"(90%: {curr_now['start_att_err_deg_p90']:.0f}) | max fitness {curr_now['fitness_max_per_drone']:.1f}",
                  flush=True)
        top = sorted(zip(CRASH_REASONS, crash_frac), key=lambda kv: -kv[1])[:3]
        print("    first-crash reasons: " + ", ".join(f"{k} {v:.0%}" for k, v in top if v > 0),
              flush=True)

    log_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()