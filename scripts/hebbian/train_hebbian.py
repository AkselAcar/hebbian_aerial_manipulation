"""
ANNOTATED VERSION of the change to train_hebbian.py -- adds 3 reps per genome.

This is not a new file to run -- it's your existing for-gen loop, with every
single line explained, plus answers to the specific things you got stuck on
in our conversation:
  - what args_cli, agents, Individual, geno2pheno actually are
  - what a "genome" is (spoiler: ABCD only, one A/B/C/D per SYNAPSE, not
    one shared rule for the whole network, and definitely not the weights)
  - what makes averaging 3 reps meaningful (the "common ground" you asked
    about) and how that's different from just having pop_size=300

Copy the annotated block below into train_hebbian.py in place of your
current for-gen loop. The comments are here to teach, not because you need
this many comments in your actual file -- feel free to trim them once it
makes sense.
=============================================================================


THE BIG PICTURE FIRST, BEFORE ANY CODE
=======================================
Every generation, two nested things happen, and they are NOT the same kind
of loop even though your code briefly uses the number 3 for both of them
(3 reps, AND 3 drones -- pure coincidence, don't let it confuse you):

  1. THE POPULATION LOOP (over pop_size, e.g. 100): this iterates over
     DIFFERENT genomes. learner.x_new is a (pop_size, 4556) array -- 100
     genuinely different ABCD vectors, sampled from CMA-ES's current search
     distribution. This is CMA-ES asking "let me try 100 different rules."

  2. THE REPS LOOP (over N_REPS, e.g. 3): for ONE fixed genome, this reruns
     the exact same ABCD rule multiple times, each time starting from a
     fresh random weight initialization. This is you asking "let me
     re-measure this SAME rule a few times, since a single measurement is
     noisy." The genome does not change across these 3 reps -- only the
     random weight init (and whatever noise the environment itself adds)
     changes. That fixed genome is exactly the "common ground" your sensor
     analogy needed.

Putting them together, you end up with a little table (this is literally
what the `fitness_reps` array holds):

                    genome 0   genome 1   genome 2   ...   genome 99
        rep 0         25.3       31.0       19.8     ...     28.7
        rep 1         28.1       29.5       33.2     ...     26.9
        rep 2         31.7       30.2       17.4     ...     27.5
                    --------------------------------------------------
        average:      28.4       30.2       23.5     ...     27.7
                     ^^^^^^^^^^ this row is fitness_reps.mean(axis=0)

Reading DOWN a column = same genome, 3 noisy measurements of it -> average
them, exactly like averaging 3 sensor readings of the same true signal.
Reading ACROSS a row = 100 different genomes, each measured once (in that
rep) -> this is the axis CMA-ES actually cares about comparing (it needs to
rank genome 0 against genome 1 against genome 2, etc.), which is exactly why
noisy single measurements per genome were the problem Fuda flagged: a noisy
row makes for an unreliable ranking. Averaging down each column first, then
handing CMA-ES the averaged row, fixes that.

This is also the direct answer to "isn't pop=300 the same as pop=100 x 3
reps?" -- no: pop=300 would give you a table with only 1 row (1 rep) and
300 columns (300 different genomes, more breadth, still each noisy). What
you're building instead is 3 rows (3 measurements) x 100 columns (100
distinct genomes) -- same total number of episodes, but aimed at making
each of the 100 column-averages trustworthy, rather than testing even more
genomes at the same noisy reliability as before.
=============================================================================
"""

import argparse
import csv
import os
import time
from isaaclab.app import AppLauncher

# --- args_cli recap -------------------------------------------------------
# args_cli is just "whatever you typed on the command line," parsed into an
# object. When your sbatch script runs:
#     train_hebbian.py --headless --pop_size=100 --n_generations=100 --episode_steps=2000
# these lines below read those flags and store them as attributes:
#     args_cli.pop_size        -> 100
#     args_cli.n_generations   -> 100
#     args_cli.episode_steps   -> 2000
# Every "args_cli.something" you see later in the script is just reading
# one of these command-line values back out.
parser = argparse.ArgumentParser(description="Hebbian/CMA-ES training for flycrane hover")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--pop_size", type=int, default=2, help="CMA-ES population size (small for smoke test)")
parser.add_argument("--n_generations", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=50, help="Steps per episode (short for smoke test)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# The ", _" throws away a second thing parse_known_args() returns: any
# command-line flags it didn't recognize (e.g. ones Isaac Sim's own launcher
# adds). We use parse_known_args() instead of parse_args() specifically so
# the script doesn't crash on those extra flags -- it just ignores them.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
import numpy as np
import torch
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.Individual import Individual, thymio_genotype
from utils.EA import CMAes

import MARL_mav_carry_ext.tasks  # noqa: F401
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg


def main():
    n_input = 45
    n_output = 5
    n_hidden = 17

    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.pop_size
    env_cfg.history_len = 1

    # The environment must be created BEFORE the check below, because the
    # check reads values from `env` (it doesn't exist until this line runs).
    env = gym.make(args_cli.task, cfg=env_cfg)

    # --- EPISODE LENGTH CHECK (NEW) ---------------------------------------
    # Compares the environment's OWN built-in time limit (max_episode_length)
    # with YOUR loop length (episode_steps). If max_episode_length is smaller
    # than episode_steps, Isaac Lab silently resets every environment mid-loop
    # and your fitness becomes a sum over several back-to-back mini-episodes.
    print("=== EPISODE LENGTH CHECK ===")
    print("env episode_length_s       :", env_cfg.episode_length_s)
    print("env step_dt (seconds/step) :", env.unwrapped.step_dt)
    print("env max_episode_length     :", env.unwrapped.max_episode_length)
    print("your episode_steps         :", args_cli.episode_steps)
    print("============================", flush=True)

    # --- agents recap -------------------------------------------------
    # env.unwrapped strips off gym's wrapper layers to get the actual
    # MARLHoverEnv object underneath. .cfg is that environment's config
    # object. .possible_agents is a list defined inside that config class,
    # holding the names of the 3 drones -- almost certainly something like
    # ["falcon1", "falcon2", "falcon3"]. So `agents` is just that list of
    # 3 strings; it has NOTHING to do with N_REPS, even though both happen
    # to be "3" in this project.
    agents = env.unwrapped.cfg.possible_agents
    device = env.unwrapped.device

    genotype = thymio_genotype("hNN", n_input, n_output)
    genotype["controller"]["params"]["n_hidden"] = n_hidden

    # --- genome size recap, and what a "genome" actually is ------------
    # n_synapses = 17*45 + 17*17 + 5*17 = 765 + 289 + 85 = 1139 connections
    # in the network. D = 4*1139 = 4556.
    # THE GENOME IS ABCD ONLY. It is NOT the network's weights. Every one
    # of the 1139 synapses gets its OWN personal A, B, C, and D coefficient
    # (not one shared A/B/C/D for the whole net) -- that's exactly why the
    # genome needs 4*1139 numbers instead of just 4. The actual weights
    # (weights_l1/l2/out) are never part of this vector; they get randomly
    # reinitialized inside every Individual(...) call below, completely
    # independent of whatever genome is being tested.
    n_synapses = n_hidden * n_input + n_hidden * n_hidden + n_output * n_hidden
    D = 4 * n_synapses

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    ea_params = {"bounds": (-5, 5), "D": D, "pop_size": args_cli.pop_size, "sigma0": 1.0}
    learner = CMAes(ea_params, output_dir="./results")
    # learner.x_new is where CMA-ES stores the CURRENT population: an array
    # shaped (pop_size, D) -- pop_size DIFFERENT genomes, each D=4556 long.
    # These are not copies of each other; they're distinct candidate rules
    # sampled from CMA-ES's current search distribution.

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    log_path = f"./logs/generation_log_{run_id}.csv"
    log_fields = ["generation", "fitness_mean", "fitness_std", "fitness_best",
                  "genome_variance_mean", "cma_sigma", "seconds"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    # =======================================================================
    # THE ACTUAL CHANGE STARTS HERE: N_REPS added around what used to be the
    # whole body of the generation loop.
    # =======================================================================
    N_REPS = 3  # per Fuda: a single episode's fitness is too noisy for CMA-ES to trust

    # --- TERMINATION DIAGNOSTICS (NEW) -----------------------------------
    # These are the 8 reasons the hover env can end an episode (see
    # _get_dones in marl_hover_env.py). The env stores each one as an
    # attribute (e.g. env.unwrapped.angle_limit_drone), a True/False value
    # per environment, recomputed on every env.step(). We read them after
    # each step and count how often each one is True, WITHOUT modifying the
    # environment code. Several can be True at once, so their counts can add
    # up to more than the total number of terminations.
    COND_NAMES = [
        "falcon_fly_low",     # a drone below 10 cm
        "payload_fly_low",    # payload below 10 cm
        "illegal_contact",    # contact force above threshold
        "angle_limit_drone",  # cable too steep relative to its drone
        "angle_limit_load",   # cable too steep relative to the payload
        "cable_collision",    # cables too close to each other
        "drone_collision",    # drones too close to each other
        "body_pos_outside",   # a drone outside the bounding box
    ]
    u = env.unwrapped  # the real env object, where those attributes live

    # --- CONTACT DIAGNOSTICS (NEW) ----------------------------------------
    # Goal: find out WHY some envs get stuck in an "illegal contact" loop.
    # Problem: by the time env.step() returns, Isaac Lab has already reset
    # the envs that terminated, which also clears their contact-sensor data.
    # So we can't inspect the crash afterwards. Instead we wrap the env's own
    # _get_dones() function: our wrapper runs the original, then takes a
    # snapshot of the contact forces and robot state BEFORE the reset happens.
    # This does not change the env's behaviour at all, it only records data.
    contact_sensor = u.scene.sensors[u.cfg.sensor_cfg.name]
    contact_body_names = list(contact_sensor.body_names)

    def take_snapshot():
        # robot state of every body: (num_envs, num_bodies, 13)
        state = u.robot.data.body_com_state_w.torch
        finite = torch.isfinite(state).reshape(state.shape[0], -1).all(dim=1)
        # contact force magnitude: (num_envs, history, num_bodies)
        forces = torch.norm(contact_sensor.data.net_forces_w_history.torch, dim=-1)
        f_per_body = forces.max(dim=1).values           # worst over history -> (num_envs, num_bodies)
        f_max, body_idx = f_per_body.max(dim=1)         # worst body -> (num_envs,)
        return {
            "finite": finite.cpu().numpy(),
            "f_max": f_max.cpu().numpy(),
            "body_idx": body_idx.cpu().numpy(),
            "drone_z_min": u.drone_positions[:, :, 2].min(dim=1).values.cpu().numpy(),
            "payload_z": u.load_position[:, 2].cpu().numpy(),
        }

    original_get_dones = u._get_dones

    def get_dones_with_snapshot():
        result = original_get_dones()      # the env's normal termination logic
        u._diag_snapshot = take_snapshot()  # record the state at this moment
        return result

    u._get_dones = get_dones_with_snapshot
    MAX_CONTACT_REPORTS = 3  # print details of the first 3 illegal contacts per env

    for gen in range(args_cli.n_generations):
        gen_start_t = time.time()

        # fitness_reps will hold the little table from the big-picture
        # explanation above: N_REPS rows, pop_size columns.
        #
        # np.zeros((N_REPS, args_cli.pop_size)) -- note this is ONE argument
        # to np.zeros, not two: the outer parentheses are the function call,
        # the inner (N_REPS, args_cli.pop_size) is a single tuple describing
        # the shape ("3 rows, pop_size columns"). np.zeros' second argument
        # would be dtype (e.g. np.float32), which we're not passing here, so
        # it defaults to float64 -- fine for fitness numbers.
        fitness_reps = np.zeros((N_REPS, args_cli.pop_size))

        # --- THE REPS LOOP: same pop_size genomes, run N_REPS separate times ---
        for rep in range(N_REPS):

            # --- build one fresh brain per genome per agent -----------
            # controllers_per_agent = {agent: [] for agent in agents} is a
            # dict comprehension -- shorthand for:
            #     controllers_per_agent = {}
            #     for agent in agents:
            #         controllers_per_agent[agent] = []
            # Result: {"falcon1": [], "falcon2": [], "falcon3": []}
            controllers_per_agent = {agent: [] for agent in agents}

            # Outer loop: over the pop_size DIFFERENT genomes in learner.x_new
            #   (this is the "different genomes" axis -- the population).
            # Inner loop: over the 3 AGENTS/drones (falcon1/2/3) -- this "3"
            #   is a completely different "3" from N_REPS=3. Don't conflate
            #   them just because they're both 3 right now.
            for genome in learner.x_new:
                for agent in agents:
                    # Individual(genotype, 0) builds ONE fresh brain: its
                    # __init__ randomly initializes weights_l1/l2/out right
                    # now, from scratch, independent of `genome`. This is
                    # the random-weight-init step that repeats every rep --
                    # the whole reason 3 reps of the SAME genome can give 3
                    # different fitness numbers.
                    #   1st arg `genotype`: the architecture TEMPLATE (how
                    #     many hidden units, states, actions) -- not the
                    #     evolved numbers.
                    #   2nd arg `0`: an id that isn't actually used for
                    #     anything downstream here -- a leftover from an
                    #     older version of this loop.
                    ind = Individual(genotype, 0)

                    # THIS is the genotype-to-phenotype step: `genome` here
                    # is ONE SPECIFIC 4556-length ABCD vector (one column
                    # in our table). geno2pheno slices it into 4 chunks of
                    # 1139 (A, B, C, D) and writes one A/B/C/D pair into
                    # EVERY synapse of ind's brain. This genome is IDENTICAL
                    # across all N_REPS reps -- only the brain's random
                    # starting weights (set above) differ each rep.
                    ind.geno2pheno(genome)

                    # File the finished, ready-to-run brain away for this
                    # drone. After both loops finish, controllers_per_agent
                    # ["falcon1"] holds pop_size controllers, one per genome.
                    controllers_per_agent[agent].append(ind.controller)

            # --- run one full episode with this rep's fresh brains -----
            obs, _ = env.reset()
            fitnesses = np.zeros(args_cli.pop_size)  # one running total per genome, THIS rep only

            # --- SPAWN DIAGNOSTICS (NEW), printed once at the very start ---
            # Shows the termination thresholds and where things are right
            # after a reset, so we can see if a condition is already
            # violated before the controller has done anything.
            if gen == 0 and rep == 0:
                print("=== SPAWN / THRESHOLD CHECK ===")
                for name in ["bounding_box_threshold", "cable_angle_limits_drone",
                             "cable_angle_limits_payload", "drone_collision_threshold",
                             "contact_sensor_threshold", "cable_collision_threshold"]:
                    print(f"cfg.{name:28s}:", getattr(u.cfg, name, "not found"))
                print("env_origins (first 2 envs):\n", u.scene.env_origins[:2].cpu().numpy())
                print("drone_positions after reset (first 2 envs):\n",
                      u.drone_positions[:2].cpu().numpy())
                print("===============================", flush=True)

            # --- TERMINATION COUNTERS (NEW) ---------------------------
            # Count how many times environments get reset by Isaac Lab
            # during this rep's episode:
            #   terminated = crash / failure triggered a reset
            #   truncated  = the env's own time limit triggered a reset
            # These can exceed pop_size if an env resets more than once.
            n_terminated = 0
            n_truncated = 0

            # NEW: per-condition counts, per-env counts, and the step at which
            # each env FIRST terminates (-1 = never terminated this rep)
            cond_counts = {name: 0 for name in COND_NAMES}
            term_per_env = np.zeros(args_cli.pop_size, dtype=int)
            first_term_step = np.full(args_cli.pop_size, -1)

            # NEW: how many illegal-contact events we've printed per env, and
            # action statistics (huge or NaN actions would point at the
            # controller rather than the simulation)
            contact_reports = np.zeros(args_cli.pop_size, dtype=int)
            act_absmax = 0.0
            n_nonfinite_act = 0

            for t in range(args_cli.episode_steps):
                actions = {}
                for agent in agents:
                    # obs[agent] has shape (pop_size, 45): row i is
                    # environment i's observation, and environment i is
                    # running genome i (they're paired up by index).
                    obs_agent = obs[agent].cpu().numpy()
                    acts = np.zeros((args_cli.pop_size, n_output), dtype=np.float32)
                    for i in range(args_cli.pop_size):
                        # controllers_per_agent[agent][i] is genome i's
                        # brain for this drone; obs_agent[i] is that same
                        # environment's current observation. Index i lines
                        # them up correctly.
                        acts[i] = controllers_per_agent[agent][i].velocity_commands(obs_agent[i])

                    # NEW: track the size of the actions and any NaN/inf values
                    finite_acts = acts[np.isfinite(acts)]
                    n_nonfinite_act += int(acts.size - finite_acts.size)
                    if finite_acts.size > 0:
                        act_absmax = max(act_absmax, float(np.abs(finite_acts).max()))

                    actions[agent] = torch.tensor(acts, dtype=torch.float32, device=device)

                obs, rewards, terminated, truncated, info = env.step(actions)
                # rewards[agents[0]] has shape (pop_size,) -- one shared
                # reward per environment (all 3 agents get the same swarm
                # reward), added onto each genome's running total.
                fitnesses += rewards[agents[0]].cpu().numpy()

                # NEW: add up how many envs were reset on this step
                n_terminated += int(terminated[agents[0]].sum().item())
                n_truncated += int(truncated[agents[0]].sum().item())

                # NEW: which envs terminated on this step, and why
                term_now = terminated[agents[0]].cpu().numpy().astype(bool).reshape(-1)
                term_per_env += term_now
                newly = term_now & (first_term_step < 0)
                first_term_step[newly] = t
                for name in COND_NAMES:
                    flag = getattr(u, name, None)
                    if flag is not None:
                        cond_counts[name] += int(flag.sum().item())

                # NEW: details of the first few illegal contacts per env, taken
                # from the snapshot recorded just before the env was reset
                snap = getattr(u, "_diag_snapshot", None)
                ic = getattr(u, "illegal_contact", None)
                if snap is not None and ic is not None:
                    ic_np = ic.cpu().numpy().astype(bool).reshape(-1)
                    for e in np.where(ic_np)[0]:
                        if e < 10 and contact_reports[e] < MAX_CONTACT_REPORTS:
                            contact_reports[e] += 1
                            print(f"rep {rep} step {t}: env {e} illegal contact #{contact_reports[e]} | "
                                  f"body={contact_body_names[snap['body_idx'][e]]} "
                                  f"force={snap['f_max'][e]:.3f} N | state finite={bool(snap['finite'][e])} | "
                                  f"lowest drone z={snap['drone_z_min'][e]:.3f} | "
                                  f"payload z={snap['payload_z'][e]:.3f}", flush=True)

            # NEW: report what happened during this rep's episode
            print(f"rep {rep}: terminated={n_terminated}, truncated={n_truncated} "
                  f"(out of {args_cli.pop_size} envs)", flush=True)
            print(f"rep {rep}: condition counts = {cond_counts}", flush=True)
            print(f"rep {rep}: terminations per env (first 10) = {term_per_env[:10].tolist()}", flush=True)
            print(f"rep {rep}: first termination step per env (first 10) = "
                  f"{first_term_step[:10].tolist()}", flush=True)
            print(f"rep {rep}: largest |action| = {act_absmax:.3f}, "
                  f"non-finite action values = {n_nonfinite_act}", flush=True)

            # NEW: one-line summary of how widespread the "stuck" problem is
            # across the WHOLE population (not just the first 10 envs):
            #   stuck_at_start = envs that terminated on the very first step
            #   likely_latched = envs with > 100 terminations in one episode
            #                    (a normal crash-and-retry gives a few dozen)
            n_stuck_start = int((first_term_step == 0).sum())
            n_latched = int((term_per_env > 100).sum())
            print(f"STUCK SUMMARY gen {gen} rep {rep}: stuck_at_start={n_stuck_start}, "
                  f"likely_latched={n_latched}, total_terminations={int(term_per_env.sum())} "
                  f"(out of {args_cli.pop_size} envs)", flush=True)

            # Store this rep's full row of pop_size fitness values into
            # row `rep` of the table.
            fitness_reps[rep] = fitnesses
        # --- end of the reps loop ---

        # Average DOWN each column: for every genome (column), average its
        # N_REPS noisy measurements into one number. axis=0 means "collapse
        # the rows" (the reps), leaving one value per column (per genome).
        # This is the line that directly answers "what does averaging
        # confirm" -- it confirms a better estimate of THAT ONE genome's
        # true quality, independently for each of the pop_size genomes.
        fitnesses = fitness_reps.mean(axis=0)

        # ===================================================================
        # EVERYTHING BELOW HERE IS UNCHANGED FROM YOUR CURRENT FILE.
        # `fitnesses` is still a length-pop_size array with the same name,
        # so nothing downstream needs to know reps happened at all.
        # ===================================================================
        fitness_mean = float(np.mean(fitnesses))
        fitness_std = float(np.std(fitnesses))
        genome_variance_mean = float(np.var(learner.x_new, axis=0).mean())
        cma_sigma = float(learner.es.sigma)
        gen_time_s = time.time() - gen_start_t

        # This is the ONLY place the actual genomes change. learner.x_new
        # gets fully replaced with a NEW set of pop_size genomes, sampled
        # from CMA-ES's updated search distribution -- this is the "CMA-ES
        # step at the end of the generation" you correctly identified.
        learner.f_new = -fitnesses  # CMA-ES minimizes; we want to maximize reward
        learner.x_new = learner.get_new_genome()
        learner.save_checkpoint()

        log_writer.writerow({
            "generation": gen,
            "fitness_mean": fitness_mean,
            "fitness_std": fitness_std,
            "fitness_best": float(np.max(fitnesses)),
            "genome_variance_mean": genome_variance_mean,
            "cma_sigma": cma_sigma,
            "seconds": gen_time_s,
        })
        log_file.flush()

        print(f"Gen {gen} done | best fitness so far = {-learner.f_best_so_far[-1]:.3f} | "
              f"fitness std = {fitness_std:.3f} | genome var = {genome_variance_mean:.4f} | "
              f"sigma = {cma_sigma:.4f} | {gen_time_s:.1f}s", flush=True)

    log_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()