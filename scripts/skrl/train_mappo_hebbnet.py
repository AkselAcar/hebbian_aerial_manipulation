"""MAPPO baseline with the Hebbian network (Fuda, 2026-11-20).

Trains MAPPO (skrl, Jack's fork) on the hover task with the same policy as the Hebbian controller:
45 inputs (history_len 1), z-normalised by skrl's RunningStandardScaler, one hidden layer of 15
(--hidden), tanh, no biases, output x [5, 5, 5, 2, 2]. One policy is shared by the 3 drones, like the
one genome per swarm in train_hebbian.py. The reward weights default to the ones of the E1 runs
(position 1.5, orientation 6, the 4 calm terms 0).

The env is built the same way as in train_hebbian.py (use_drone, history_len 1, weights set on the cfg),
so no change to the task registration is needed. The agent config is
exts/.../hover/agents/skrl_mappo_hebbnet_cfg.yaml.

Afterwards, scripts/skrl/export_mappo_to_hebbian.py turns a checkpoint into start weights + frozen
z-statistics for train_hebbian.py --from_mappo.

  train_mappo_hebbnet.py --headless --num_envs 4096 --drone falcon --tag falcon_lr3e-4 --policy_lr 3e-4

Output: logs/skrl/mappo_hebbnet/<date>_<tag>/ (checkpoints/agent_<step>.pt, params/, hebbnet_args.json)
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MAPPO with the Hebbian network shape")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--drone", type=str, default="falcon", help="falcon or omninxt (controllers/drone_params.py)")
parser.add_argument("--robot_usd", type=str, default=None)
parser.add_argument("--hidden", type=int, nargs="+", default=[15], help="hidden layer sizes (Hebbian E1: 15)")
parser.add_argument("--pos_weight", type=float, default=1.5)
parser.add_argument("--ori_weight", type=float, default=6.0)
parser.add_argument("--calm_weight", type=float, default=0.0,
                    help="weight of each of the 4 calm-flight terms (env default 0.5, E1 runs 0)")
parser.add_argument("--policy_lr", type=float, default=None, help="policy learning rate (cfg: 5e-5)")
parser.add_argument("--value_lr", type=float, default=None, help="value learning rate (cfg: 1e-4)")
parser.add_argument("--timesteps", type=int, default=None, help="env steps per drone (cfg: 1.5M)")
parser.add_argument("--checkpoint_interval", type=int, default=None, help="steps between checkpoints (cfg: 25000)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="resume from this agent_<step>.pt (policy, critic, optimizers and input statistics); "
                         "the step count starts again from 0 in a new log folder")
parser.add_argument("--tag", type=str, default="", help="appended to the log folder name")
parser.add_argument("--cfg", type=str, default=None, help="agent yaml (default: agents/skrl_mappo_hebbnet_cfg.yaml)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime

import gymnasium as gym
import skrl
import yaml
from skrl.utils.runner.torch import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import MARL_mav_carry_ext.tasks  # noqa: F401
import MARL_mav_carry_ext.tasks.directMARL.hover.agents as hover_agents
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg

CALM_TERMS = ("action_smoothness_weight", "body_rate_penalty_weight", "force_penalty_weight", "downwash_rew_weight")
N_INPUT, N_OUTPUT = 45, 5
OUTPUT_SCALE = [5.0, 5.0, 5.0, 2.0, 2.0]


def main():
    # --- agent config -------------------------------------------------------
    cfg_path = args_cli.cfg or os.path.join(os.path.dirname(hover_agents.__file__), "skrl_mappo_hebbnet_cfg.yaml")
    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)
    if not agent_cfg["models"].get("shared_across_agents", False):
        print("WARNING: the policy is not shared across the drones; the export will use falcon1's", flush=True)
    net = agent_cfg["models"]["policy"]["network"][0]
    net["layers"] = [{"linear": {"out_features": h, "bias": False}} for h in args_cli.hidden] + \
                    [{"linear": {"out_features": N_OUTPUT, "bias": False}}]
    net["activations"] = ["tanh"] * len(net["layers"])
    if args_cli.policy_lr is not None:
        agent_cfg["agent"]["learning_rate"][0] = args_cli.policy_lr
    if args_cli.value_lr is not None:
        agent_cfg["agent"]["learning_rate"][1] = args_cli.value_lr
    if args_cli.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
    if args_cli.checkpoint_interval is not None:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = args_cli.checkpoint_interval
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["seed"] = args_cli.seed

    # --- env config, as in train_hebbian.py ---------------------------------
    env_cfg = MARLHoverEnvCfg()
    env_cfg.use_drone(args_cli.drone, args_cli.robot_usd)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.history_len = 1
    # the class computed the spaces with history_len 3 (135); skrl sizes its memory from them
    env_cfg.observation_spaces = {a: N_INPUT for a in env_cfg.possible_agents}
    env_cfg.pos_track_weight = args_cli.pos_weight
    env_cfg.ori_track_weight = args_cli.ori_weight
    for k in CALM_TERMS:
        setattr(env_cfg, k, args_cli.calm_weight)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # --- log folder ----------------------------------------------------------
    root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        name += f"_{job}"
    if args_cli.tag:
        name += f"_{args_cli.tag}"
    agent_cfg["agent"]["experiment"]["directory"] = root
    agent_cfg["agent"]["experiment"]["experiment_name"] = name
    log_dir = os.path.join(root, name)
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # what the export script needs to rebuild the same controller
    meta = {**vars(args_cli), "n_input": N_INPUT, "n_output": N_OUTPUT, "output_scale": OUTPUT_SCALE,
            "skrl_version": skrl.__version__, "cfg_path": cfg_path}
    with open(os.path.join(log_dir, "hebbnet_args.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)

    print(f"[mappo_hebbnet] drone {env_cfg.drone} | network {N_INPUT} -> {' -> '.join(map(str, args_cli.hidden))} -> "
          f"{N_OUTPUT} (tanh, no bias, x {OUTPUT_SCALE}) | weights pos {env_cfg.pos_track_weight} "
          f"ori {env_cfg.ori_track_weight} calm {args_cli.calm_weight} | lr {agent_cfg['agent']['learning_rate']} | "
          f"timesteps {agent_cfg['trainer']['timesteps']} | log {log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
        print(f"[mappo_hebbnet] resumed from {args_cli.checkpoint}", flush=True)
    pol = runner.agent.policies[env.possible_agents[0]]
    print("[mappo_hebbnet] policy parameters:",
          {k: tuple(v.shape) for k, v in pol.state_dict().items()}, flush=True)
    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()