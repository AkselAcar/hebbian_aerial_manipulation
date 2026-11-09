#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-2
#SBATCH -J diag_replays
#SBATCH -o logs/%x_%A_%a.out

# Three diagnostic replays of a trained controller, 64 flights each (32 swarms x 2 episodes), no video:
#   task 0: normal, records the reward per term   -> replay_theta_<stamp>
#   task 1: plasticity switched off after 5 s     -> replay_theta_freeze5s_<stamp>   (weight-drift hypothesis)
#   task 2: drone ID inputs set to 0              -> replay_theta_noid_<stamp>       (does it rely on the ID?)
#
#   sbatch jobs/diag_replays.sh                          # best run (default)
#   sbatch jobs/diag_replays.sh results/<other run>
# Then: python3 diagnose_replay.py --replay <task 0 folder> <task 1 folder> <task 2 folder> --out diag

RUN=${1:-results/3219114_s0.05_eslr0.01_BEST}
EXTRA=("" "--freeze_after 5" "--zero_id")
i=$SLURM_ARRAY_TASK_ID
echo "task $i: $RUN ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_diag$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 --lr 0.001 ${EXTRA[$i]}
  "