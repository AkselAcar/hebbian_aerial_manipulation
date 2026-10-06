#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH -J hebbian_cmaes_pop100_reps3_overnight
#SBATCH -o logs/%x_%j.out

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

# n_generations is set high on purpose: the 8 h time limit stops the job
# (about 45-50 generations). The CSV log and the CMA-ES checkpoint are saved
# after every generation, so nothing is lost when the time limit is reached.
apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless --pop_size=100 --n_generations=200 --episode_steps=2000
  "