"""
Hebbian / evolution-strategy training for the flycrane hover task (version 5).

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


def flat(x):
    """Env outputs can be (n_envs,) or (n_envs, 1): make them (n_envs,)."""
    return x.reshape(-1)


def main():
    n_input, n_hidden, n_output = 45, 17, 5

    pop = args_cli.pop_size
    n_reps = args_cli.n_reps
    n_envs = pop * n_reps  # one environment per (genome, rep) pair
    T = args_cli.episode_steps

    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = n_envs
    env_cfg.history_len = 1
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

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    if args_cli.tag:
        run_id = f"{run_id}_{args_cli.tag}"   # e.g. generation_log_3217000_eslr0.001_s0.02.csv
    results_dir = f"./results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    ea_params = {"bounds": (-5, 5), "D": D, "pop_size": pop,
                 "sigma0": args_cli.sigma0, "x0": x0,
                 "es_lr": args_cli.es_lr, "weight_decay": args_cli.weight_decay,
                 "es_optimizer": args_cli.es_optimizer}
    if args_cli.optimizer == "cmaes":
        learner = CMAes(ea_params, output_dir=results_dir)
    else:
        learner = OpenAIES(ea_params, output_dir=results_dir)
    print(f"optimizer={args_cli.optimizer}, es_optimizer={args_cli.es_optimizer}, "
          f"es_lr={args_cli.es_lr}, weight_decay={args_cli.weight_decay}")
    print(f"genome={args_cli.genome}, D={D}, sigma0={args_cli.sigma0}, "
          f"lr={args_cli.lr}, results in {results_dir}", flush=True)

    brains = BatchedHebbian(n_input, n_hidden, n_output, lr=args_cli.lr, device=device)
    scale = brains.action_scale                                   # [5, 5, 5, 2, 2]

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
    ] + [f"crash_{name}" for name in CRASH_REASONS]           # share of crashed swarms
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

        obs, _ = env.reset()
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
            acts = brains.act(obs_all, update_mask=mask)       # crashed swarms: weights frozen
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
            **{f"crash_{name}": float(crash_frac[k]) for k, name in enumerate(CRASH_REASONS)},
        })
        log_file.flush()

        print(f"Gen {gen} done | fitness mean {fitness_mean:.3f} best {fitness_best:.3f} "
              f"(x3: {n_agents * fitness_best:.1f}) | survived mean {np.mean(survived):.0f} "
              f"best {np.max(survived):.0f} steps | full episodes {frac_full:.1%} | "
              f"stuck_at_start {stuck_at_start} | sim steps {steps_simulated} | "
              f"sat {sat_frac:.1%} | |w| {w_abs_start:.3f}->{w_abs_end:.3f} | "
              f"cmd acc {act_acc_frac:.1%} rate {act_rate_frac:.1%} of limit | "
              f"sigma {sigma:.3f} | {gen_time_s:.1f}s", flush=True)
        top = sorted(zip(CRASH_REASONS, crash_frac), key=lambda kv: -kv[1])[:3]
        print("    first-crash reasons: " + ", ".join(f"{k} {v:.0%}" for k, v in top if v > 0),
              flush=True)

    log_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()