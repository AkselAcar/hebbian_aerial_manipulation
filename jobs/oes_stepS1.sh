#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH -J oes_stepS1
#SBATCH -o logs/%x_%A_%a.out

# Step S1: static baseline for the robustness tests. Exactly the recipe of the best Hebbian run
# (E1_fuda_omninxt: OmniNxt, no curriculum, 8 reps, pop 256, ori weight 6, z-norm, 45 -> 15 -> 5, 700 gens),
# but the genome is the weights only (--genome=w): no plasticity at all, evolved from scratch.
# Fair comparison "Hebbian vs evolving the weights directly": unlike switching plasticity off in a Hebbian
# controller, this network was trained to work without plasticity.
#   task 0 g700:   700 generations  -> compare with E1_fuda_omninxt (same recipe, 700 gens)
#   task 1 g1200: 1200 generations  -> compare with E2_long_omninxt (same recipe, 1200 gens)

GENS=(700 1200)
i=$SLURM_ARRAY_TASK_ID
G=${GENS[$i]}
echo "task $i: static, $G generations"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_S1_$i
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
      --genome=w --drone=omninxt --n_reps=8 --pop_size=256 --ori_weight_start=6 --obs_norm=z --hidden 15 \
      --calm_weight=0 --collision_off_gens=300 --collision_ramp_gens=200 \
      --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 \
      --lr_schedule=cosine --lr_min_frac=0.1 --n_generations=$G --lr_total_gens=$G --tag=S1_static_g${G}_omninxt
  "