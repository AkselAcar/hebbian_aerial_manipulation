#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-6
#SBATCH -J checks_replays
#SBATCH -o logs/%x_%A_%a.out

# Checks from the Nov 20 meeting, for one run (its drone, its network, its plasticity settings):
#   task 0 hold:     the network flies, the goal is the payload's start pose (zero error) -> can it stay there?
#   task 1 zero:     every drone commands 0 (no network) -> what "doing nothing" looks like
#   task 2 pd:       hand-made controller acc = 2 * goal error - 2 * velocity (no network) -> can the task be
#                    solved through this action interface at all? (position only)
#   task 3 step_z:   scripted +1 / -1 m/s^2 up            \
#   task 4 sine_x:   scripted 1 m/s^2 sine in x at 0.5 Hz  > low-level tracking, no network
#   task 5 rate_p:   scripted roll-rate steps +-0.5 rad/s  /
#   task 6 video:    3 normal episodes of one swarm with video (Newton viewer)
# Same seed for all: the same starts (and goals) everywhere.
#   sbatch jobs/checks_replays.sh results/<run>
# Output: results/<run>/replay_<label>_<stamp>/ ; then (plain python3, repo root):
#   python3 check_tracking.py --replay results/<run>/replay_signal_step_z_*      (also sine_x, rate_p, and the
#                                                    normal or hold replays: is the network's own output tracked?)
#   python3 diagnose_replay.py --replay results/<run>/replay_theta_goalatstart_* results/<run>/replay_signal_pd_* \
#       results/<run>/replay_zero_action_*  --out diag_checks

RUN=$1
[ -f "$RUN/theta.npy" ] || { echo "usage: sbatch jobs/checks_replays.sh results/<run>"; exit 1; }
i=$SLURM_ARRAY_TASK_ID
COMMON="--results_dir $RUN --which theta --lr 0.001"
ARGS=("--headless $COMMON --num_envs 32 --goal_at_start"
      "--headless $COMMON --num_envs 32 --zero_action"
      "--headless $COMMON --num_envs 32 --test_signal pd"
      "--headless $COMMON --num_envs 8 --test_signal step_z"
      "--headless $COMMON --num_envs 8 --test_signal sine_x"
      "--headless $COMMON --num_envs 8 --test_signal rate_p"
      "$COMMON --num_envs 1 --n_episodes 3 --video --viz newton")
echo "task $i: ${ARGS[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_checks$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/replay_hebbian.py ${ARGS[$i]}
  "