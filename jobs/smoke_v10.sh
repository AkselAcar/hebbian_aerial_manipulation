#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J smoke_v10
#SBATCH -o logs/%x_%A_%a.out

# Quick check of train_hebbian v8.4 + the new env files on the OLD drone (~10 min): 3 tiny generations with the
# weight ramps (orientation 3.0 -> 1.5, position 0.5 -> 1.5 over 2 gens), decay and the mean gate.
# Look in the .out for:
#   "drone model: falcon"
#   "[bodies]" line without WARNING
#   task: "pos w 0.50, ori w 3.00" -> "1.00 / 2.25" -> "1.50 / 1.50"
#   "rotors: at the thrust limit ...% of the time, mean thrust ..% of max" (numbers, not nan)
#   no Traceback.

TAG=smoke10
EXTRA=("--calm_weight=0 --hebb_decay=0.2 --hebb_gate=1.0 --hebb_gate_att=45 --hebb_gate_mode=mean --ori_weight_start=3.0 --ori_weight_end=1.5 --ori_weight_gens=2 --pos_weight_start=0.5 --pos_weight_end=1.5 --pos_weight_gens=2")
i=0
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke10
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