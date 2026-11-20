#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH -J oes_stepE1
#SBATCH -o logs/%x_%A_%a.out

# Step E1 (2026-11-20): Fuda's changes, all at once, from scratch, with Jack's cable-angle fix in the env,
# on the old drone (Falcon, to compare with C1) and on the new one (OmniNxt):
#   8 evaluations per genome (n_reps 8; population 256 to keep 2048 parallel swarms), orientation weight 6
#   (position 1.5), z-normalised inputs (scripts/hebbian/zstats.json), network 45 -> 15 -> 5.
# Kept from our recipe: regularization terms at 0, collision termination off for 300 gens, Hebbian decay 0.2 /s.
#   task 0 fuda_falcon:       old drone, nothing else
#   task 1 fuda_curr_falcon:  old drone + orientation curriculum (heading 15 -> 180 deg over gens 0-420,
#                             tilt over 420-560); compare with C1 orient
#   task 2 fuda_omninxt:      new drone, nothing else
#   task 3 fuda_curr_omninxt: new drone + orientation curriculum
# 700 generations (about 48 h if a generation takes ~240 s with 2048 swarms; check the "s" at the end of the Gen lines).
# Logs: logs/oes_stepE1_<A>_<task>.out

COMMON="--n_reps=8 --pop_size=256 --ori_weight_start=6 --obs_norm=z --hidden 15 --calm_weight=0 --collision_off_gens=300 --collision_ramp_gens=200 --hebb_decay=0.2"
CURR="--curr_yaw_gens=420 --curr_tilt_gens=140"
NAMES=(fuda_falcon fuda_curr_falcon fuda_omninxt fuda_curr_omninxt)
EXTRA=("$COMMON --drone=falcon"
       "$COMMON --drone=falcon $CURR"
       "$COMMON --drone=omninxt"
       "$COMMON --drone=omninxt $CURR")
i=$SLURM_ARRAY_TASK_ID
TAG="E1_${NAMES[$i]}"
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_E1_$i
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
      --n_generations=700 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=700 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "