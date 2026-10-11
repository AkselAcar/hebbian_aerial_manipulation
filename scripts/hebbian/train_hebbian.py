"""
Hebbian / CMA-ES training for the flycrane hover task (version 2).

Changes compared with version 1 (the first commit on this branch):

1. ALIVE MASK: reward only counts until a swarm's FIRST termination
   (crash, illegal contact, ...). After that the env is auto-reset by Isaac
   Lab, but its reward is ignored. All reward terms in this env are
   positive, so surviving longer always gives a higher fitness.

2. PARALLEL REPS: all N_REPS evaluations of every genome run at the same
   time, in pop_size * N_REPS environments. Environment j runs genome
   j % pop_size (rep = j // pop_size). Each env has its own freshly
   initialized brains, so the reps stay independent. The episode also
   stops early as soon as every env has terminated once.

3. LOGGING: the CSV now also contains steps survived (= how long swarms
   fly before their first crash), the fraction of evaluations that
   survived the full episode, and fitness on the "summed over 3 drones"
   scale used by skrl / Jack / Fuda (= 3 x the per-drone fitness).

Fitness scales: per drone, the maximum for a perfect 2000-step episode is
100 (5 reward terms weight sum 5 x step_dt 0.01 x 2000 steps). Summed over
the 3 drones the maximum is 300.
"""

import argparse
import csv
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hebbian/CMA-ES training for flycrane hover (v2)")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--pop_size", type=int, default=2, help="CMA-ES population size")
parser.add_argument("--n_reps", type=int, default=3, help="Evaluations per genome (run in parallel)")
parser.add_argument("--n_generations", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=50, help="Max steps per episode")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.EA import CMAes
from utils.Individual import Individual, thymio_genotype

import MARL_mav_carry_ext.tasks  # noqa: F401
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg


def main():
    n_input = 45
    n_output = 5
    n_hidden = 17

    pop = args_cli.pop_size
    n_reps = args_cli.n_reps
    n_envs = pop * n_reps  # one environment per (genome, rep) pair

    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = n_envs
    env_cfg.history_len = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    print("=== SETUP ===")
    print(f"pop_size={pop}, n_reps={n_reps} -> num_envs={n_envs}")
    print(f"env max_episode_length={u.max_episode_length}, your episode_steps={args_cli.episode_steps}")
    print("=============", flush=True)

    agents = u.cfg.possible_agents
    n_agents = len(agents)
    device = u.device

    genotype = thymio_genotype("hNN", n_input, n_output)
    genotype["controller"]["params"]["n_hidden"] = n_hidden
    n_synapses = n_hidden * n_input + n_hidden * n_hidden + n_output * n_hidden
    D = 4 * n_synapses  # 4556 ABCD coefficients

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    ea_params = {"bounds": (-5, 5), "D": D, "pop_size": pop, "sigma0": 1.0}
    learner = CMAes(ea_params, output_dir="./results")

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    log_path = f"./logs/generation_log_{run_id}.csv"
    log_fields = [
        "generation",
        "fitness_mean", "fitness_std", "fitness_best",        # per drone (max 100)
        "fitness_mean_3drones", "fitness_best_3drones",       # summed scale (max 300)
        "steps_survived_mean", "steps_survived_best",         # steps until first crash
        "frac_full_episode",                                  # evaluations that never crashed
        "stuck_at_start",                                     # envs terminating on step 0
        "steps_simulated",                                    # episode length actually run
        "genome_variance_mean", "cma_sigma", "seconds",
    ]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    for gen in range(args_cli.n_generations):
        gen_start_t = time.time()

        # --- one fresh brain per environment and drone ----------------------
        # Environment j runs genome j % pop. The three envs that share a genome
        # (j, j + pop, j + 2*pop) each get their own Individual, so their
        # random starting weights and Hebbian updates are independent.
        controllers = {agent: [] for agent in agents}
        for j in range(n_envs):
            genome = learner.x_new[j % pop]
            for agent in agents:
                ind = Individual(genotype, 0)
                ind.geno2pheno(genome)
                controllers[agent].append(ind.controller)

        obs, _ = env.reset()
        fitness_env = np.zeros(n_envs)                         # reward until first termination
        alive = np.ones(n_envs, dtype=bool)                    # False after first termination
        steps_survived = np.full(n_envs, args_cli.episode_steps)
        stuck_at_start = 0
        steps_simulated = 0

        for t in range(args_cli.episode_steps):
            alive_idx = np.where(alive)[0]
            actions = {}
            for agent in agents:
                obs_agent = obs[agent].cpu().numpy()
                acts = np.zeros((n_envs, n_output), dtype=np.float32)
                # Only alive envs need their controller evaluated; dead envs
                # get zero actions (their reward is ignored anyway).
                for j in alive_idx:
                    acts[j] = controllers[agent][j].velocity_commands(obs_agent[j])
                actions[agent] = torch.tensor(acts, dtype=torch.float32, device=device)

            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_simulated = t + 1

            r = rewards[agents[0]].cpu().numpy().reshape(-1)
            fitness_env += r * alive  # the step on which a swarm crashes still counts

            crashed = terminated[agents[0]].cpu().numpy().astype(bool).reshape(-1)
            timed_out = truncated[agents[0]].cpu().numpy().astype(bool).reshape(-1)

            new_crash = crashed & alive
            steps_survived[new_crash] = t + 1
            if t == 0:
                stuck_at_start = int(new_crash.sum())

            alive &= ~(crashed | timed_out) # Once an env has terminated, it is no longer alive and its reward is ignored.
            if not alive.any():
                break  # every env has terminated once: nothing left to measure

        # --- average the reps of each genome --------------------------------
        # Env j = rep * pop + i, so reshaping to (n_reps, pop) puts rep in the
        # row and genome i in the column.
        fitness_reps = fitness_env.reshape(n_reps, pop)
        survived_reps = steps_survived.reshape(n_reps, pop)
        fitnesses = fitness_reps.mean(axis=0)
        survived = survived_reps.mean(axis=0)

        fitness_mean = float(np.mean(fitnesses))
        fitness_std = float(np.std(fitnesses))
        fitness_best = float(np.max(fitnesses))
        genome_variance_mean = float(np.var(learner.x_new, axis=0).mean())
        cma_sigma = float(learner.es.sigma)

        learner.f_new = -fitnesses  # CMA-ES minimizes; we maximize reward
        learner.x_new = learner.get_new_genome()
        learner.save_checkpoint()

        gen_time_s = time.time() - gen_start_t
        frac_full = float(np.mean(steps_survived >= args_cli.episode_steps))

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
            "steps_simulated": steps_simulated,
            "genome_variance_mean": genome_variance_mean,
            "cma_sigma": cma_sigma,
            "seconds": gen_time_s,
        })
        log_file.flush()

        print(f"Gen {gen} done | fitness mean {fitness_mean:.3f} best {fitness_best:.3f} "
              f"(x3: {n_agents * fitness_best:.1f}) | survived mean {np.mean(survived):.0f} "
              f"best {np.max(survived):.0f} steps | full episodes {frac_full:.1%} | "
              f"stuck_at_start {stuck_at_start} | sim steps {steps_simulated} | "
              f"sigma {cma_sigma:.3f} | {gen_time_s:.1f}s", flush=True)

    log_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()