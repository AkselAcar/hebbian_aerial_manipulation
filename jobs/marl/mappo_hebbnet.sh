#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH -J mappo_hebbnet
#SBATCH -o logs/%x_%A_%a.out

# MAPPO baseline with the Hebbian network (Fuda, 2026-11-20): policy 45 -> 15 -> 5, tanh, no biases,
# output x [5, 5, 5, 2, 2], inputs z-normalised by skrl's running scaler, one policy shared by the 3 drones.
# Same task as E1: position weight 1.5, orientation 6, calm terms 0, env with Jack's cable-angle fix.
# Critic and PPO settings as Jack's skrl_mappo_cfg.yaml, 4096 envs, 1.5M steps.
#   task 0 falcon_lr5e-5:   old drone, Jack's policy lr
#   task 1 falcon_lr3e-4:   old drone, higher policy lr (the policy is ~1000x smaller than Jack's)
#   task 2 omninxt_lr5e-5:  new drone
#   task 3 omninxt_lr3e-4
# Runs in logs/skrl/mappo_hebbnet/<date>_<job>_<task name>/ (TensorBoard there, checkpoints every 25k steps).
# Then: scripts/skrl/export_mappo_to_hebbian.py --checkpoint <run>/checkpoints/best_agent.pt

NAMES=(falcon_lr5e-5 falcon_lr3e-4 omninxt_lr5e-5 omninxt_lr3e-4)
EXTRA=("--drone=falcon --policy_lr=5e-5"
       "--drone=falcon --policy_lr=3e-4"
       "--drone=omninxt --policy_lr=5e-5"
       "--drone=omninxt --policy_lr=3e-4")
i=$SLURM_ARRAY_TASK_ID
echo "task $i: ${NAMES[$i]} ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SKRL=$HOME/lis_semester_project/skrl
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_mappo_$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1
grep -q shared_across_agents $SKRL/skrl/multi_agents/torch/mappo/mappo_cfg.py \
  || { echo "Jack's skrl fork in $SKRL has no parameter sharing: run 'cd $SKRL && git pull' first"; exit 1; }

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B "$SKRL":/workspace/algos/skrl \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:/workspace/algos/skrl:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train_mappo_hebbnet.py --headless --num_envs=4096 \
      --ori_weight=6 --pos_weight=1.5 --calm_weight=0 ${EXTRA[$i]} --tag=${NAMES[$i]}
  "