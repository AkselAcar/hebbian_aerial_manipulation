#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=debug
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J isaac_smoke
#SBATCH -o logs/%x_%j.out

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SKRL=$HOME/lis_semester_project/skrl
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

apptainer exec --nv \
  --env ACCEPT_EULA=Y \
  -B "$REPO":/workspace/project \
  -B "$SKRL":/workspace/algos/skrl \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:/workspace/algos/skrl:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/skrl/train.py \
      --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
      --algorithm=MAPPO \
      --headless \
      --num_envs=32 \
      --max_iterations=3
  "