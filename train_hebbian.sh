#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J hebbian_gen1_pop2_reps3_test_file
#SBATCH -o logs/%x_%j.out

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian_crash_test.py --headless --pop_size=2 --n_generations=1 --episode_steps=2000
  "