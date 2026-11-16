#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J hover_test
#SBATCH -o logs/%x_%j.out

# Hover test: every drone gets action 0 (no acceleration, no body rate) for 20 s; the network is not used.
# Question: does the payload hang tilted by itself (centre of mass / attachment geometry), which would
# explain the ~25-30 deg tilt every trained controller ends with?
# 32 swarms, normal random starts (payload starts level, random heading).
#
#   sbatch jobs/hover_test.sh
#
# Output: results/3219114_s0.05_eslr0.01_BEST/replay_zero_action_<stamp>/ (ep0.npz, summary.txt)
# In the .out: "payload tilt (median over alive swarms) at 0 s: .. deg, 1 s: .., ..., 20 s: .."
# (swarms that drift out or crash drop out of the median.)

RUN=results/3219114_s0.05_eslr0.01_BEST   # only used to build the network; its output is ignored
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_hover
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --zero_action --num_envs 32 --n_episodes 1
  "