#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH -J oes_stepD1
#SBATCH -o logs/%x_%A_%a.out

# Step D1 (2026-11-18): first runs on the NEW DRONE (OmniNxt, 6 N per rotor), reward = position + orientation,
# orientation curriculum (heading 15 -> 180 deg over gens 0-600, tilt 0 -> 45 deg over 600-800), Hebbian decay
# 0.2 /s, collision termination off for 300 gens, cosine lr. The recipe that worked from scratch in C1.
#   task 0 orient:      from scratch (= C1 orient + decay, on the new drone)
#   task 1 orient_surv: + survival signal: smoothness, body rate, downwash 0.5 -> 0 over the first 300 gens,
#                         force penalty 0 throughout
#   task 2 orient_warm: warm start from 3219114 (the old controller, which already flies the OmniNxt)
#   task 3 orient_pw2:  + position weight 1.5 -> 2.0 over 500 gens (middle ratio between C1 orient and pw3)
# Needs train_hebbian v8.5 and the built OmniNxt USD. Logs: logs/oes_stepD1_<A>_<task>.out
# Watch "rotors: at the thrust limit" in the "full flights" lines.

COMMON="--drone=omninxt --collision_off_gens=300 --collision_ramp_gens=200 --hebb_decay=0.2 --curr_yaw_gens=600 --curr_tilt_gens=200"
NOCALM="--calm_weight=0"
NAMES=(orient orient_surv orient_warm orient_pw2)
EXTRA=("$COMMON $NOCALM"
       "$COMMON --calm_weight=0 --calm_weight_start=0.5 --calm_fade_gens=300 --force_weight=0"
       "$COMMON $NOCALM --init_from=/workspace/project/results/3219114_s0.05_eslr0.01_BEST/theta.npy"
       "$COMMON $NOCALM --pos_weight_end=2.0 --pos_weight_gens=500")
i=$SLURM_ARRAY_TASK_ID
TAG="D1_${NAMES[$i]}"
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_D1_$i
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
      --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=1000 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "