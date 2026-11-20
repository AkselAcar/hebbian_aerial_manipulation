#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J smoke_v12
#SBATCH -o logs/%x_%A_%a.out

# Quick check of the E1 setup (~10 min): 2 tiny generations with Fuda's changes on the OmniNxt
# (z-normalised inputs, network 45 -> 15 -> 5, orientation weight 6) and the env with Jack's fix.
# Look in the .out for:
#   "network 45 -> 15 -> 5 (750 synapses, genome 3750), inputs: z-normalised"
#   "drone model: omninxt", task lines with "ori w 6.00", and no Traceback.

TAG=smoke12
EXTRA=("--drone=omninxt --ori_weight_start=6 --obs_norm=z --hidden 15 --calm_weight=0 --hebb_decay=0.2")
i=0
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke12
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
      --pop_size=8 --n_reps=1 --n_generations=2 --episode_steps=300 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=2 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "