#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -J slide_replay
#SBATCH -o logs/%x_%j.out

# 64 test flights (32 swarms x 2 episodes, no rendering) of a trained controller, recording
# everything slide_figures.py needs: positions, payload attitude error and rotation, goal
# rotation, payload velocities, and the weights of all swarms (--record_weights).
#
#   sbatch jobs/slide_replay.sh                         # best run (default)
#   sbatch jobs/slide_replay.sh results/<other run>
#
# Output: <run>/replay_theta_<timestamp>/   ep0.npz ep1.npz weights_ep0.npz weights_ep1.npz summary.txt
# Then:   python3 slide_figures.py --replay <that folder> --out slide_figs [--video]

RUN=${1:-results/3219114_s0.05_eslr0.01_BEST}
echo "slide replay of $RUN"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_slides
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 --lr 0.001 --record_weights
  "
