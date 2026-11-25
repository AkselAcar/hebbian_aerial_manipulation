#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J drone_checks
#SBATCH -o logs/%x_%j.out

# Checks after building the two USDs (jobs/build_flycrane_asset.sh falcon / omninxt), ~40 min:
#   1. pipeline check: 3219114 on the RE-CONVERTED Falcon USD, same 64 flights (seed 0) as review_seed.
#      Must match its review_seed row (97% full, 0.55 m, 18% within 0.3 m) up to small noise.
#   2. hover test on the OmniNxt: zero actions, 32 swarms -> payload tilt over time, crashes.
#   3. 3219114 flown on the OmniNxt (no retraining): how far the old controller transfers.
#   4. tiny training on the OmniNxt (2 generations): the training path works with the new drone.
# Look in the .out for "[bodies]" (drone 1-3, rotors grouped per drone, no WARNING), "[drone]" (start rotor
# speeds, max thrust), "payload tilt", "full episodes", "sat" (thrust saturation), and no Traceback.

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_drone_checks
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1
RUN=results/3219114_s0.05_eslr0.01_BEST
FALCON_USD=/workspace/project/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR/flycrane_falcon/flycrane_falcon/flycrane_falcon.usda

run() {
  apptainer exec --nv \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
    -B "$REPO":/workspace/project \
    -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
    "$SIF" \
    /bin/bash -c "
      export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
      cd /workspace/project
      /workspace/isaaclab/_isaac_sim/python.sh $*
    "
}

ONLY=${1:-all}    # sbatch jobs/drone_checks.sh 1   -> only test 1
echo "=== 1. pipeline check: Falcon, re-converted USD ==="
run scripts/hebbian/replay_hebbian.py --headless --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 \
    --lr 0.001 --drone falcon --robot_usd $FALCON_USD
[ "$ONLY" = 1 ] && exit 0
echo "=== 2. hover test, OmniNxt ==="
run scripts/hebbian/replay_hebbian.py --headless --results_dir $RUN --zero_action --num_envs 32 --n_episodes 1 \
    --drone omninxt
[ "$ONLY" = 1 ] && exit 0
echo "=== 3. 3219114 on the OmniNxt ==="
run scripts/hebbian/replay_hebbian.py --headless --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 \
    --lr 0.001 --drone omninxt
[ "$ONLY" = 1 ] && exit 0
echo "=== 4. tiny training, OmniNxt ==="
run scripts/hebbian/train_hebbian.py --headless --pop_size=8 --n_reps=1 --n_generations=2 --episode_steps=300 \
    --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 \
    --lr=0.001 --init_from=/workspace/project/$RUN/theta.npy --drone omninxt --tag=smoke_omninxt