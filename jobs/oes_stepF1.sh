#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH -J oes_stepF1
#SBATCH -o logs/%x_%j.out

# Step F1: Hebbian A, B, C, D evolved on top of a MAPPO policy (Fuda, 2026-11-20).
# The start weights are the MAPPO weights and stay fixed; the inputs use MAPPO's z-statistics, frozen.
# Generation 0 (A = B = C = D = 0) is the MAPPO policy itself, so the curve shows what plasticity adds.
# Same task as the MAPPO run (pos 1.5, ori 6, calm 0) and as E1 (8 reps, pop 256, decay 0.2 /s).
# No curricula and no collision-off phase: the MAPPO policy already flies.
#
#   sbatch jobs/oes_stepF1.sh results/mappo_<run>_best_agent [tag]
# The drone is the one MAPPO was trained on (from the export's train_args.json).

EXP=$1
TAG=${2:-F1_$(basename "$EXP")}
[ -f "$EXP/fixed_weights.npy" ] || { echo "usage: sbatch jobs/oes_stepF1.sh results/mappo_<run>_<ckpt> [tag]"; exit 1; }
echo "from MAPPO export $EXP, tag $TAG"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_F1_$SLURM_JOB_ID
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/train_hebbian.py --headless --from_mappo $EXP \
      --n_generations=400 --episode_steps=2000 --n_reps=8 --pop_size=256 \
      --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=400 --lr_min_frac=0.1 \
      --ori_weight_start=6 --calm_weight=0 --hebb_decay=0.2 --tag=$TAG
  "