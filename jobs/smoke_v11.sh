#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J smoke_v11
#SBATCH -o logs/%x_%A_%a.out

# Quick check of train_hebbian v8.5 on the NEW DRONE (~10 min): 3 tiny generations with D1's options
# (OmniNxt, curriculum, decay, survival fade 0.5 -> 0 over 2 gens, force weight 0).
# Look in the .out for:
#   "drone model: omninxt" and "[drone] omninxt: ... max thrust 6.00 N per rotor"
#   task: "calm w 0.50, force w 0.00" -> "0.25" -> "0.00", heading change growing
#   "rotors: at the thrust limit ...", and no Traceback.

TAG=smoke11
EXTRA=("--drone=omninxt --hebb_decay=0.2 --curr_yaw_gens=2 --curr_tilt_gens=1 --calm_weight=0 --calm_weight_start=0.5 --calm_fade_gens=2 --force_weight=0")
i=0
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke11
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
      --pop_size=8 --n_reps=1 --n_generations=3 --episode_steps=300 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=3 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "