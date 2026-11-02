#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-13%7
#SBATCH -J mass_test
#SBATCH -o logs/%x_%A_%a.out

# Robustness test of a TRAINED controller (no training): replay its final genome theta
# with different payload masses. It was trained at 1.4 kg only.
#   7 masses x {plasticity on (Hebbian lr 0.001), plasticity off (lr 0, weights frozen)}
#   = 14 tasks, 7 at a time (%7), ~5 min each. Each: 32 swarms x 2 episodes, no video.
#
#   sbatch jobs/mass_test.sh                                   # best run (default)
#   sbatch jobs/mass_test.sh results/<other run>
#
# Output: <run>/replay_theta_m<mass>[_heblr0]_<timestamp>/  (summary.txt + ep*.npz)
# Plot:   python3 plot_mass_test.py --run <run>

RUN=${1:-results/3219114_s0.05_eslr0.01_BEST}
MASSES=(2.4 2.8 3.2 3.6 4.0 0.6 0.8)
HEB_LRS=(0.001 0)

i=$SLURM_ARRAY_TASK_ID
MASS=${MASSES[$((i % 7))]}
HEB_LR=${HEB_LRS[$((i / 7))]}
echo "task $i: run=$RUN payload=$MASS kg hebbian_lr=$HEB_LR"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
# one cache folder per task: parallel Isaac Sim instances must not share a Kit data store
KIT=/scratch/izar/$USER/kit_task$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 --lr $HEB_LR --payload_mass $MASS
  "