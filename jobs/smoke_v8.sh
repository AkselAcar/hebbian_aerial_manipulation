#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J smoke_v8
#SBATCH -o logs/%x_%j.out

# Quick check of the v8 curricula before launching the A2 runs (~10 min):
# 4 tiny generations (8 genomes x 1 rep, 300 steps) with every step-2 option on, on a 2-generation schedule.
# Look in the .out, on the "task:" line under each "Gen ... done" (Gen 0 -> Gen 3):
#   heading change <= 15 -> 188 -> 360 -> 360 deg     tilt <= 0 -> 0 -> 0 -> 45 deg
#   z +-0.10 -> 0.30 -> 0.50 -> 0.50 m                 pos w 1.50 -> 2.25 -> 3.00 -> 3.00
#   calm w 0.10 every gen                              collision 0.00 -> 0.00 -> 0.60 -> 0.60 m
# and "asked at start: ... deg" small at Gen 0 and large from Gen 2 on (= the env really uses the new ranges).

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

run() {
  apptainer exec --nv \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
    -B "$REPO":/workspace/project \
    -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
    "$SIF" \
    /bin/bash -c "
      export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
      /workspace/isaaclab/_isaac_sim/python.sh $*
    "
}

echo "=== tiny training with all curricula ==="
run /workspace/project/scripts/hebbian/train_hebbian.py --headless \
  --pop_size=8 --n_reps=1 --n_generations=4 --episode_steps=300 \
  --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 --lr=0.001 \
  --init_from=/workspace/project/results/3219114_s0.05_eslr0.01_BEST/theta.npy \
  --curr_yaw_gens=2 --curr_tilt_gens=1 --curr_height_gens=2 --pos_weight_end=3.0 --pos_weight_gens=2 \
  --calm_weight=0.1 --collision_off_gens=1 --collision_ramp_gens=1 --tag=smoke8