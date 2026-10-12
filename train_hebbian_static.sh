#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=09:30:00
#SBATCH -J hebbian_cmaes_static_w_p100_r3_s0.1
#SBATCH -o logs/%x_%j.out

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

# Static baseline: CMA-ES evolves only the network weights (genome 1139),
# A = B = C = D = 0, so the weights never change during an episode.
# Same settings as job 3215535 (weights + ABCD) so the two runs are comparable.
# Alive mask + parallel reps (300 envs = 100 genomes x 3 reps).
# n_generations is set high on purpose: the time limit stops the job.
apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless --pop_size=100 --n_reps=3 --n_generations=2000 --episode_steps=2000 --genome=w --sigma0=0.1 --lr=0.001
  "