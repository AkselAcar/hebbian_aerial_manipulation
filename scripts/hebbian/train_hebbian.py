"""Evolve Hebbian controllers for the flycrane task (3 drones carrying a payload) with an evolution strategy.

Each generation:
  1. the optimizer (OpenAI-ES or CMA-ES, utils/EA.py) proposes pop_size genomes;
  2. every genome is flown n_reps times; all swarms run in parallel in Isaac Lab, and each drone gets
     its own Hebbian network built from the genome (utils/batched_hebbian.py);
  3. a swarm's fitness is the reward summed over the episode, up to its first termination (crash);
     the reps of a genome are averaged;
  4. the fitnesses go back to the optimizer, which makes the next population.

Genome (--genome): w_abcd = [start weights | A | B | C | D], 5 x n values (n = 1139 synapses for the
                   default 17-17 network, 750 for --hidden 15); w = start weights only, a static network.
                   --from_mappo <folder>: start weights fixed to a MAPPO policy (export_mappo_to_hebbian.py),
                   only [A | B | C | D] evolve (4 x n), inputs z-normalised with MAPPO's frozen statistics.
Fitness scale: per drone, a perfect 2000-step episode gives (sum of the reward weights) x 0.01 s x 2000,
i.e. 100 with the default weights (6 terms summing to 5); the logs also give it as fitness_max_per_drone.

Options, grouped (all off / at the env's values unless given):
  optimizer       --optimizer, --pop_size, --n_reps, --sigma0, --es_lr, --es_optimizer, --weight_decay,
                  --lr_schedule cosine (with --lr_total_gens, --lr_min_frac), --init_from (warm start)
  network         --hidden (layer sizes, default 17 17), --obs_norm fixed|z with --zstats (input scaling)
  plasticity      --lr (Hebbian learning rate), --hebb_decay (pull back to the start weights, 1/s),
                  --hebb_gate / --hebb_gate_att / --hebb_gate_mode (scale the update by the position
                  and/or attitude error)
  task            --curr_yaw_gens / --curr_tilt_gens / --curr_height_gens (curricula that start easy and
                  end on the normal task), --collision_off_gens, --mass_range, --drone falcon|omninxt
  reward weights  --pos_weight_start/_end, --ori_weight_start/_end, --calm_weight, --calm_weight_start
                  with --calm_fade_gens, --force_weight

Outputs: logs/generation_log_<run>.csv (one row per generation), and results/<run>/ with the optimizer
state (theta.npy = the OpenAI-ES centre, the controller we evaluate; x_best.npy), train_args.json and
theta_hist/theta_gen<N>.npy (the centre every --save_theta_every generations, for replays along training).

"""

import argparse
import csv
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hebbian/ES training for the flycrane hover task")
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
parser.add_argument("--hidden", type=int, nargs="+", default=[17, 17],
                    help="hidden layer sizes, e.g. --hidden 15 for one layer of 15 (default: 17 17)")
parser.add_argument("--obs_norm", choices=["fixed", "z"], default="fixed",
                    help="input scaling: fixed bounds (default) or z-normalisation with the statistics of --zstats")
parser.add_argument("--zstats", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "zstats.json"),
                    help="json file with the per-input mean and std used by --obs_norm z")
parser.add_argument("--mass_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="payload mass range in kg, new mass every reset (default: fixed 1.4 kg)")
parser.add_argument("--save_theta_every", type=int, default=25,
                    help="OpenAI-ES: save the search centre every N generations in results/<run>/theta_hist/ "
                         "(replay one with replay_hebbian.py --gen N; 0 = off)")
parser.add_argument("--from_mappo", type=str, default=None,
                    help="folder written by scripts/skrl/export_mappo_to_hebbian.py: its weights become the fixed "
                         "start weights, only A, B, C, D evolve (genome 4 x n, zeros = the MAPPO policy itself); "
                         "sets --hidden, --obs_norm z and --zstats from that folder")
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
parser.add_argument("--hebb_gate_att", type=float, default=0.0,
                    help="neuromodulation on the orientation too: attitude error scale [deg] (0 = off)")
parser.add_argument("--hebb_gate_mode", choices=["max", "mean"], default="max",
                    help="how the position and attitude terms are combined (see plasticity_gate)")
# --- curricula and reward weights (all off by default) --------------------------------------
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
parser.add_argument("--pos_weight_start", type=float, default=None,
                    help="position reward weight at generation 0 (default: env value 1.5)")
parser.add_argument("--ori_weight_start", type=float, default=None,
                    help="orientation reward weight at generation 0 (default: env value 1.5). Alone = constant: "
                         "--ori_weight_start=0 is the position-only run")
parser.add_argument("--ori_weight_end", type=float, default=None,
                    help="orientation reward weight after --ori_weight_gens (default: stays at the start value)")
parser.add_argument("--ori_weight_gens", type=int, default=500, help="generations for the orientation weight ramp")
parser.add_argument("--calm_weight_start", type=float, default=None,
                    help="survival signal: smoothness, body rate and downwash weights start at this value and fade "
                         "linearly to --calm_weight (default 0.5 if not given) over --calm_fade_gens generations")
parser.add_argument("--calm_fade_gens", type=int, default=300, help="generations of that fade")
parser.add_argument("--force_weight", type=float, default=None,
                    help="force (thrust effort) penalty weight, set separately from the other calm terms (constant)")
parser.add_argument("--calm_weight", type=float, default=None,
                    help="weight of each calm-flight term (smoothness, body rate, force, downwash; env: 0.5)")
parser.add_argument("--collision_off_gens", type=int, default=0,
                    help="no drone-collision termination for the first N gens, then switched back on gradually")
parser.add_argument("--collision_ramp_gens", type=int, default=200,
                    help="generations over which the collision distance goes from 0 back to the env value (0.6 m)")
parser.add_argument("--drone", type=str, default=None,
                    help="drone model, controllers/drone_params.py: falcon (default) or omninxt")
parser.add_argument("--robot_usd", type=str, default=None,
                    help="flycrane USD to load instead of the drone's default one (pipeline checks)")
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


def plasticity_gate(obs_all, gate_pos, gate_att, mode="max"):
    """Neuromodulation of the Hebbian update, per drone (None = no gate).

    Each drone sees the payload's goal errors in its own observation:
      position error  |obs[33:36]| [m]                -> e_p = error / gate_pos
      attitude error  angle of the matrix obs[36:45]  -> e_a = error / gate_att [deg]
    m = min(1, e)  with  e = max(e_p, e_a)  ("keep adapting while either error is large")
                    or   e = mean(e_p, e_a) (mode "mean"); a term with gate 0 is left out.
    """
    terms = []
    if gate_pos > 0:
        terms.append(obs_all[:, 33:36].norm(dim=1) / gate_pos)
    if gate_att > 0:
        R = obs_all[:, 36:45].reshape(-1, 3, 3)
        cos = ((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)
        terms.append(torch.rad2deg(torch.arccos(cos)) / gate_att)
    if not terms:
        return None
    e = terms[0] if len(terms) == 1 else (torch.maximum(*terms) if mode == "max" else 0.5 * (terms[0] + terms[1]))
    return e.clamp(max=1.0)


def ramp(gen, start, n, a, b):
    """a before generation `start`, b from `start + n` on, linear in between."""
    if gen < start:
        return a
    if n <= 0 or gen >= start + n:
        return b
    return a + (b - a) * (gen - start) / n


CALM_TERMS = ("action_smoothness_weight", "body_rate_penalty_weight", "force_penalty_weight", "downwash_rew_weight")


class Curriculum:
    """Changes the task before each generation's env.reset().

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
                         pos_w=float(u.cfg.pos_track_weight), ori_w=float(u.cfg.ori_track_weight),
                         coll=float(u.cfg.drone_collision_threshold))
        # reward weight ramps: start (default env value) -> end (default: position stays, orientation stays)
        self.pos_w = (args.pos_weight_start if args.pos_weight_start is not None else self.full["pos_w"],
                      args.pos_weight_end)
        ori0 = args.ori_weight_start if args.ori_weight_start is not None else self.full["ori_w"]
        self.ori_w = (ori0, args.ori_weight_end if args.ori_weight_end is not None else ori0)
        self.weights_changed = (args.pos_weight_start is not None or args.pos_weight_end is not None
                                or args.ori_weight_start is not None or args.ori_weight_end is not None)
        if args.calm_weight is not None:
            for k in CALM_TERMS:
                setattr(u.cfg, k, args.calm_weight)
        if args.force_weight is not None:
            u.cfg.force_penalty_weight = args.force_weight
        # fade of the 3 non-force calm terms: calm_weight_start -> calm_weight (env 0.5 if not given)
        self.calm_fade = None
        if args.calm_weight_start is not None:
            end = args.calm_weight if args.calm_weight is not None else 0.5
            self.calm_fade = (args.calm_weight_start, end)
        self.active = any([args.curr_yaw_gens, args.curr_tilt_gens, args.curr_height_gens,
                           self.weights_changed, args.calm_weight is not None, args.collision_off_gens,
                           args.calm_weight_start is not None, args.force_weight is not None])

    @staticmethod
    def _sym(full, half):
        """Symmetric range +-half around the centre of `full`, never wider than `full`."""
        mid, fh = 0.5 * float(full[0] + full[1]), 0.5 * float(full[1] - full[0])
        h = min(float(half), fh)
        return (float(mid - h), float(mid + h))   # plain floats: numpy float64 makes Isaac's reset tensors float64 -> crash

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
        p0, p1 = self.pos_w
        u.cfg.pos_track_weight = float(ramp(gen, 0, a.pos_weight_gens, p0, p1) if p1 is not None else p0)
        u.cfg.ori_track_weight = float(ramp(gen, 0, a.ori_weight_gens, *self.ori_w))
        s["pos_weight_now"] = float(u.cfg.pos_track_weight)
        s["ori_weight_now"] = float(u.cfg.ori_track_weight)
        if self.calm_fade is not None:
            w = float(ramp(gen, 0, a.calm_fade_gens, *self.calm_fade))
            for k in CALM_TERMS:
                if k != "force_penalty_weight" or a.force_weight is None:
                    setattr(u.cfg, k, w)
        s["calm_weight_now"] = float(getattr(u.cfg, CALM_TERMS[0]))
        s["force_weight_now"] = float(u.cfg.force_penalty_weight)
        # drone collision
        if a.collision_off_gens:
            u.cfg.drone_collision_threshold = float(ramp(gen, a.collision_off_gens, a.collision_ramp_gens, 0.0, f["coll"]))
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


def pose_errors(obs0):
    """Payload-goal distance [m] and attitude error [deg] of every env, from drone 0's observation."""
    pos_err = obs0[:, 33:36].norm(dim=1)
    R = obs0[:, 36:45].reshape(-1, 3, 3)                      # goal rotation x payload rotation^T
    cos = ((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)
    return pos_err, torch.rad2deg(torch.arccos(cos))


def start_errors(obs0):
    """What the reset actually asked for (from drone 0's observation): start distance to the goal,
    start attitude error and start payload height. Used to check that the curriculum takes effect."""
    pos_err, att = pose_errors(obs0)
    z = obs0[:, 2]
    return {"start_pos_err_mean": float(pos_err.mean()), "start_att_err_deg_mean": float(att.mean()),
            "start_att_err_deg_p90": float(torch.quantile(att.float(), 0.9)),
            "start_payload_z_min": float(z.min()), "start_payload_z_max": float(z.max())}


CURR_FIELDS = ["curr_heading_change_max_deg", "curr_tilt_max_deg", "curr_z_half_range",
               "pos_weight_now", "ori_weight_now", "calm_weight_now", "force_weight_now", "collision_dist_now", "fitness_max_per_drone",
               # rotors (alive swarms): share of rotor-time at the thrust limit, mean thrust / limit over all
               # rotors, and the same for the most loaded drone of each swarm
               "rotor_sat_frac", "rotor_thrust_frac", "rotor_thrust_frac_max_drone",
               "start_pos_err_mean", "start_att_err_deg_mean", "start_att_err_deg_p90",
               "start_payload_z_min", "start_payload_z_max",
               # learning curves of the actual task, over the flights that lasted the whole episode:
               # error at the end (last state before the final step) and averaged over the flight
               "pos_err_end_full", "att_err_end_deg_full", "pos_err_mean_full", "att_err_mean_deg_full"]


def main():
    n_input, n_output = 45, 5
    fixed_w = None                           # start weights that are not evolved (--from_mappo)
    if args_cli.from_mappo:
        with open(os.path.join(args_cli.from_mappo, "train_args.json")) as f:
            m_args = json.load(f)
        fixed_w = np.load(os.path.join(args_cli.from_mappo, "fixed_weights.npy")).astype(float).reshape(-1)
        if list(args_cli.hidden) != list(m_args["hidden"]):
            print(f"--from_mappo: hidden layers {args_cli.hidden} -> {m_args['hidden']} (the MAPPO network)", flush=True)
        args_cli.hidden = list(m_args["hidden"])
        args_cli.obs_norm = "z"
        args_cli.zstats = os.path.join(args_cli.from_mappo, "zstats.json")   # frozen MAPPO statistics
        args_cli.genome = "abcd"
        if args_cli.drone is None:
            args_cli.drone = m_args.get("drone")
        elif args_cli.drone != m_args.get("drone"):
            print(f"WARNING: MAPPO was trained on {m_args.get('drone')}, running on {args_cli.drone}", flush=True)
    hidden = tuple(args_cli.hidden)

    pop = args_cli.pop_size
    n_reps = args_cli.n_reps
    n_envs = pop * n_reps  # one environment per (genome, rep) pair
    T = args_cli.episode_steps

    env_cfg = MARLHoverEnvCfg()
    args_cli.drone = args_cli.drone or "falcon"            # saved in train_args.json
    env_cfg.use_drone(args_cli.drone, args_cli.robot_usd)
    print(f"drone model: {env_cfg.drone} | USD {env_cfg.robot_cfg.spawn.usd_path}", flush=True)
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

    sizes = (n_input,) + hidden + (n_output,)
    n_synapses = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))  # 1139 for 17-17, 750 for 15
    D = {"w_abcd": 5, "abcd": 4, "w": 1}[args_cli.genome] * n_synapses           # 5695 / 4556 / 1139 for 17-17
    if fixed_w is not None and fixed_w.size != n_synapses:
        raise ValueError(f"--from_mappo has {fixed_w.size} weights, the network needs {n_synapses}")
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

    obs_mean = obs_std = None
    if args_cli.obs_norm == "z":
        with open(args_cli.zstats) as f:
            zs = json.load(f)
        obs_mean, obs_std = zs["mean"], zs["std"]
        with open(os.path.join(results_dir, "zstats.json"), "w") as f:   # kept with the run, read by the replay
            json.dump(zs, f, indent=1)
    if fixed_w is not None:
        np.save(os.path.join(results_dir, "fixed_weights.npy"), fixed_w)    # read by the replay
        print(f"start weights fixed from {args_cli.from_mappo} (mean |w| {np.abs(fixed_w).mean():.3f}); "
              f"evolving A, B, C, D only", flush=True)
    brains = BatchedHebbian(n_states=n_input, n_actions=n_output, lr=args_cli.lr, device=device,
                            decay=args_cli.hebb_decay * u.step_dt,       # per-second rate -> per step
                            hidden=hidden, obs_mean=obs_mean, obs_std=obs_std)
    print(f"network {' -> '.join(map(str, sizes))} ({n_synapses} synapses, genome {D}), "
          f"inputs: {'z-normalised (' + args_cli.zstats + ')' if obs_mean is not None else 'fixed bounds'}", flush=True)
    print(f"lr schedule {args_cli.lr_schedule} (total {args_cli.lr_total_gens} gens, min {args_cli.lr_min_frac} x es_lr) | "
          f"plasticity brakes: decay {args_cli.hebb_decay}/s, gate {args_cli.hebb_gate} m, "
          f"{args_cli.hebb_gate_att} deg ({args_cli.hebb_gate_mode})", flush=True)
    with open(os.path.join(results_dir, "train_args.json"), "w") as f:
        json.dump({**vars(args_cli), "step_dt": u.step_dt, "D": D}, f, indent=1, default=str)
    scale = brains.action_scale                                   # [5, 5, 5, 2, 2]
    curr = Curriculum(u, args_cli)
    if curr.active:
        print(f"curricula: heading {args_cli.curr_yaw_gens} gens (from {args_cli.curr_yaw0} deg), "
              f"tilt {args_cli.curr_tilt_gens} gens, height {args_cli.curr_height_gens} gens (from +-{args_cli.curr_z0} m), "
              f"pos weight {curr.pos_w[0]} -> {curr.pos_w[1]} over {args_cli.pos_weight_gens} gens, "
              f"ori weight {curr.ori_w[0]} -> {curr.ori_w[1]} over {args_cli.ori_weight_gens} gens, "
              f"calm weight {args_cli.calm_weight} (fade from {args_cli.calm_weight_start} over {args_cli.calm_fade_gens} gens), "
              f"force weight {args_cli.force_weight}, collision off {args_cli.collision_off_gens} gens "
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
    ] + CURR_FIELDS + [f"crash_{name}" for name in CRASH_REASONS]   # task / reward values, share of crashed swarms
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    for gen in range(args_cli.n_generations):
        gen_start_t = time.time()
        k = args_cli.save_theta_every
        if k and hasattr(learner, "theta") and (gen % k == 0 or gen == args_cli.n_generations - 1):
            os.makedirs(os.path.join(results_dir, "theta_hist"), exist_ok=True)   # centre that generates this gen
            np.save(os.path.join(results_dir, "theta_hist", f"theta_gen{gen:04d}.npy"), learner.theta)

        # --- one brain per (drone, env), all built from the population at once --
        # Env j runs genome j % pop: its 3 drones and its reps start from the same
        # evolved weights and rules, but their Hebbian updates are independent.
        genomes = learner.x_new
        if fixed_w is not None:                                      # [MAPPO weights | evolved A, B, C, D]
            genomes = np.concatenate([np.broadcast_to(fixed_w, (genomes.shape[0], n_synapses)), genomes], 1)
        brains.load(genomes, n_envs, n_agents)
        w_abs_start = brains.mean_abs_w1(agent=0)

        curr_now = curr.apply(gen)       # set this generation's task and reward weights (no-op when all off)
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
        pos_last = att_last = None
        pos_sum = torch.zeros(n_envs, device=device)                  # sum over steps of the pose errors
        att_sum = torch.zeros(n_envs, device=device)
        n_crashed = torch.zeros((), device=device)
        rotor_sums = torch.zeros(3, device=device)                    # see CURR_FIELDS (rotor_*)
        alive_steps = torch.zeros((), device=device)

        for t in range(T):
            # payload pose error of the state the drones see now (valid for envs still alive)
            pe, ae = pose_errors(obs[agents[0]])
            pos_last = torch.where(alive, pe, pos_last) if t else pe.clone()
            att_last = torch.where(alive, ae, att_last) if t else ae.clone()
            pos_sum += pe * alive; att_sum += ae * alive
            # all drones of all envs in one batch: brain b = agent * n_envs + env
            obs_all = torch.cat([obs[a] for a in agents], dim=0)
            mask = alive.repeat(n_agents)
            gate = plasticity_gate(obs_all, args_cli.hebb_gate, args_cli.hebb_gate_att, args_cli.hebb_gate_mode)
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
            if hasattr(u, "rotor_sat_now"):                      # env with the rotor saturation read-out
                af = alive.float()
                rotor_sums += torch.stack([(u.rotor_sat_now * af).sum(), (u.rotor_thrust_frac_now * af).sum(),
                                           (u.rotor_thrust_frac_max_drone * af).sum()])
                alive_steps += af.sum()

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
        full = steps_survived >= T                                     # flights without any termination
        nf = max(int(full.sum()), 1)
        n_seen = max(steps_simulated, 1)
        curr_now.update({"pos_err_end_full": float(pos_last[full].sum()) / nf if full.any() else np.nan,
                         "att_err_end_deg_full": float(att_last[full].sum()) / nf if full.any() else np.nan,
                         "pos_err_mean_full": float(pos_sum[full].sum()) / (nf * n_seen) if full.any() else np.nan,
                         "att_err_mean_deg_full": float(att_sum[full].sum()) / (nf * n_seen) if full.any() else np.nan})
        rot = (rotor_sums / alive_steps.clamp(min=1)).cpu().numpy() if float(alive_steps) > 0 else [np.nan] * 3
        curr_now.update({"rotor_sat_frac": float(rot[0]), "rotor_thrust_frac": float(rot[1]),
                         "rotor_thrust_frac_max_drone": float(rot[2])})
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
                  f"pos w {curr_now['pos_weight_now']:.2f}, ori w {curr_now['ori_weight_now']:.2f}, calm w {curr_now['calm_weight_now']:.2f}, force w {curr_now['force_weight_now']:.2f}, "
                  f"collision {curr_now['collision_dist_now']:.2f} m | asked at start: "
                  f"{curr_now['start_pos_err_mean']:.2f} m, {curr_now['start_att_err_deg_mean']:.0f} deg "
                  f"(90%: {curr_now['start_att_err_deg_p90']:.0f}) | max fitness {curr_now['fitness_max_per_drone']:.1f}",
                  flush=True)
        print(f"    full flights: end error {curr_now['pos_err_end_full']:.2f} m / {curr_now['att_err_end_deg_full']:.0f} deg, "
              f"mean over flight {curr_now['pos_err_mean_full']:.2f} m / {curr_now['att_err_mean_deg_full']:.0f} deg | "
              f"rotors: at the thrust limit {curr_now['rotor_sat_frac']:.1%} of the time, mean thrust "
              f"{curr_now['rotor_thrust_frac']:.0%} of max, most loaded drone {curr_now['rotor_thrust_frac_max_drone']:.0%}",
              flush=True)
        top = sorted(zip(CRASH_REASONS, crash_frac), key=lambda kv: -kv[1])[:3]
        print("    first-crash reasons: " + ", ".join(f"{k} {v:.0%}" for k, v in top if v > 0),
              flush=True)

    log_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()