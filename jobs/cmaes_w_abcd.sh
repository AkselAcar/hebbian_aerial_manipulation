#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=09:30:00
#SBATCH -J hebbian_cmaes_w_abcd_p100_r3_s0.1_lr1e-3
#SBATCH -o logs/%x_%j.out

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

# Quick test of the evolved-weights setup: 2 genomes x 3 reps = 6 envs, 1 generation.
# Check in the .out file:
#   1. setup line: genome=w_abcd, D=5695, sigma0=0.1, lr=0.001
#   2. "sat" close to 0%
#   3. "survived mean" well above ~50 steps
apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless --pop_size=100 --n_reps=3 --n_generations=2000 --episode_steps=2000 --genome=w_abcd --sigma0=0.1 --lr=0.001
  "
  