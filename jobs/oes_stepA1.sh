#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH -J oes_stepA1
#SBATCH -o logs/%x_%A_%a.out

# Step A1 (2026-11): plasticity brakes, warm start from the best controller (3219114).
# Same settings as 3219114 (pop 512 x 2 reps, sigma 0.05, Adam 0.01, weights + ABCD, Hebbian lr 0.001),
# plus a cosine learning-rate decay 0.01 -> 0.001 over 1000 generations in every task:
#   task 0: control (cosine only)
#   task 1: + weight decay towards the start weights, 0.2 /s (time constant 5 s)
#   task 2: + error-gated plasticity, update x min(1, |goal error| / 1.0 m)
# Stops by itself after 1000 generations or 48 h.
# Logs: logs/oes_stepA1_<A>_<task>.out   CSVs: logs/generation_log_<jobid>_A1_<name>.csv
# Compare afterwards with diag replays (jobs/diag_replays.sh <results folder>) + diagnose_replay.py,
# NOT by fitness alone.

NAMES=(cos cos_decay0.2 cos_gate1.0 const)
EXTRA=("" "--hebb_decay=0.2" "--hebb_gate=1.0" "--lr_schedule=const")
i=$SLURM_ARRAY_TASK_ID
TAG="A1_${NAMES[$i]}"
INIT=results/3219114_s0.05_eslr0.01_BEST/theta.npy
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_A1_$i
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