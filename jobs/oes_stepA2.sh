#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH -J oes_stepA2
#SBATCH -o logs/%x_%A_%a.out

# Step A2 (2026-11): curricula and reward weights, warm start from the best controller (3219114).
# Same settings as A1 task 0 (cosine lr 0.01 -> 0.001, no brakes), so A1 task 0 is the control.
#   task 0 orient:   heading change asked grows 15 -> 180 deg over 400 gens, then goal tilt 0 -> 45 deg over 300
#   task 1 height:   start/goal height 1 +- 0.1 m -> 0.5-1.5 m over 400 gens; position weight 1.5 -> 3.0 over 500
#   task 2 calm0.1:  smoothness, body rate, force, downwash weights 0.5 -> 0.1 each (whole run)
#   task 3 posw3:    position weight 1.5 -> 3.0 over 500 gens alone (separates it from the height curriculum of task 1)
# (the collision-off curriculum is kept for the from-scratch runs of step 3)
# Every curriculum ends on the normal task. Fitness is NOT comparable between tasks (other weights):
# compare with diag replays (jobs/diag_replays.sh <results folder>) + diagnose_replay.py.
# Logs: logs/oes_stepA2_<A>_<task>.out   CSVs: logs/generation_log_<jobid>_A2_<name>.csv

NAMES=(orient height calm0.1 posw3)
EXTRA=("--curr_yaw_gens=400 --curr_tilt_gens=300"
       "--curr_height_gens=400 --pos_weight_end=3.0 --pos_weight_gens=500"
       "--calm_weight=0.1"
       "--pos_weight_end=3.0 --pos_weight_gens=500")
i=$SLURM_ARRAY_TASK_ID
TAG="A2_${NAMES[$i]}"
INIT=results/3219114_s0.05_eslr0.01_BEST/theta.npy
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_A2_$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/train_hebbian.py --headless \
      --pop_size=512 --n_reps=2 --n_generations=1000 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 --init_from=/workspace/project/$INIT \
      --lr_schedule=cosine --lr_total_gens=1000 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "