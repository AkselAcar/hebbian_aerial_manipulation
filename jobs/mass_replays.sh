#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH -J mass_replays
#SBATCH -o logs/%x_%A_%a.out

# Out-of-distribution test 1 (Fuda, Nov 23): payload mass. Every controller was trained with 1.4 kg only.
# Each controller flies the exam flights (same seed, 64 flights) with the payload at 0.7 ... 2.2 kg.
#   CTRL 0  E1 Hebbian (best, OmniNxt)        plasticity on, as trained
#   CTRL 1  E1 Hebbian, plasticity OFF         same weights, Hebbian lr 0 -> what does plasticity add?
#   CTRL 2  MAPPO (OmniNxt, lr 5e-5)          static network
# Task i -> controller i / 5, mass i % 5.  Then: python3 plot_mass_sweep.py --replays results/*/replay_*_m* --out mass_sweep

RUNS=(results/3262390_E1_fuda_omninxt results/3262390_E1_fuda_omninxt results/mappo_omninxt_lr5e-5)
HLR=(0.001 0 0.001)
MASSES=(0.7 1.0 1.4 1.8 2.2)
i=$SLURM_ARRAY_TASK_ID
C=$((i / 5)); M=${MASSES[$((i % 5))]}
RUN=${RUNS[$C]}
echo "task $i: $RUN, Hebbian lr ${HLR[$C]}, payload $M kg"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_mass$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 --lr ${HLR[$C]} --payload_mass $M
  "