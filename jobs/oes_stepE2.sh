#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH -J oes_stepE2
#SBATCH -o logs/%x_%A_%a.out

# Step E2: can the E1 recipe (Fuda's changes) get further on the real drone (OmniNxt)?
# E1 plateaued: its ES learning rate had decayed to the minimum by gen 700, and with the curriculum the heading
# error rose from ~7 deg to ~37 deg exactly while the asked turn widened (gens 0-420) -> the task got harder
# faster than the network learned. E1 without curriculum ended best on heading (~33 deg).
#   task 0 long:      E1 recipe, no curriculum, 1200 gens (cosine over 1200)          -> does more time help?
#   task 1 slowcurr:  E1 recipe, heading curriculum over 800 gens (was 420), tilt over 800-950, 1200 gens
#   task 2 continue:  warm start from E1_fuda_omninxt's final controller, fresh ES learning rate, 400 gens
# Same as E1 otherwise (8 reps, pop 256, ori weight 6, z-norm, 45 -> 15 -> 5, decay 0.2, calm terms 0).
# 1200 gens ~ 69 h at E1's ~206 s per generation (qos normal allows 72 h).

E1_PREV=results/3262390_E1_fuda_omninxt     # task 2 starts from its theta.npy
COMMON="--drone=omninxt --n_reps=8 --pop_size=256 --ori_weight_start=6 --obs_norm=z --hidden 15 --calm_weight=0 --hebb_decay=0.2"
NAMES=(long slowcurr continue)
EXTRA=("$COMMON --collision_off_gens=300 --collision_ramp_gens=200 --n_generations=1200 --lr_total_gens=1200"
       "$COMMON --collision_off_gens=300 --collision_ramp_gens=200 --n_generations=1200 --lr_total_gens=1200 --curr_yaw_gens=800 --curr_tilt_gens=150"
       "$COMMON --init_from=$E1_PREV/theta.npy --n_generations=400 --lr_total_gens=400")
i=$SLURM_ARRAY_TASK_ID
TAG="E2_${NAMES[$i]}_omninxt"
[ $i -ne 2 ] || [ -f "$E1_PREV/theta.npy" ] || { echo "task 2: $E1_PREV/theta.npy not found, fix E1_PREV"; exit 1; }
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_E2_$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/train_hebbian.py --headless --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 --lr_schedule=cosine --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "