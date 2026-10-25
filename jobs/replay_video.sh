#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J replay_hebbian
#SBATCH -o logs/%x_%j.out

# Replay a trained controller and record what the swarms do.
#
#   sbatch replay_hebbian.sh <results folder> [theta | best] [generation]
#
#   theta          = final OpenAI-ES centre (theta.npy). The controller the run produced. Default.
#   best           = best single sample (x_best.npy). Add a generation to get the best
#                    sample found up to that generation (default: end of the run).
#
# Examples:
#   sbatch replay_hebbian.sh results/3217661_p512_eslr0.01                 # final controller
#   sbatch replay_hebbian.sh results/3217661_p512_eslr0.01 best 400        # how it flew at gen 400
#
# Two steps:
#   1. 32 swarms x 2 episodes, no rendering -> summary.txt + trajectories (ep*.npz)
#   2. 1 swarm x 6 episodes with an .mp4 each (Newton viewer, works on the V100)
# Output: <results folder>/replay_<theta|best_gen<N>>_<timestamp>/  (one folder per step)
# Plot afterwards (login node, no GPU): python3 plot_replay.py --replay <that folder>

RUN=${1:-results/3217661_p512_eslr0.01}
WHICH=${2:-theta}
GEN=${3:--1}
echo "replaying $WHICH (generation $GEN) of $RUN"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

# Writable Isaac Sim caches (Fuda's guide, section 1.4). Without them Kit can fail with
# "Failed to create local file data store" -- the error that crashed sweep 2's task 1.
KIT=/scratch/izar/$USER/kit        # izar's scratch is /scratch/izar/<user>, not /scratch/<user>
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

run() {
  apptainer exec --nv \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
    -B "$REPO":/workspace/project \
    -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
    "$SIF" \
    /bin/bash -c "
      export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
      /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/replay_hebbian.py $*
    "
}

echo "=== step 1: statistics + trajectories (no rendering) ==="
run --headless --results_dir $RUN --which $WHICH --best_gen $GEN --num_envs 32 --n_episodes 2 --lr 0.001

echo "=== step 2: videos of single swarms (Newton viewer) ==="
#run --results_dir $RUN --which $WHICH --best_gen $GEN --num_envs 1 --n_episodes 6 --lr 0.001 --video --viz newton