#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH -J smoke_v9
#SBATCH -o logs/%x_%j.out

# Quick check of train_hebbian v8.2 / replay_hebbian (orientation gate + hover test), ~10 min:
#   1. 2 tiny generations with the position + orientation gate, warm start from the gate run
#   2. tiny replay of that result (must read the gate back from train_args.json)
#   3. tiny hover test (zero actions)
# Look in the .out for:
#   "plasticity brakes: decay 0.0/s, gate 1.0 m, 45.0 deg (max)"   twice (training and replay)
#   "Gen 0 done", "Gen 1 done", "saved replay to" twice, "payload tilt (median ...)" lines, no Traceback.

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke9
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

INIT=/workspace/project/results/3240622_A1_cos_gate1.0/theta.npy

echo "=== 1. tiny training ==="
run /workspace/project/scripts/hebbian/train_hebbian.py --headless \
  --pop_size=8 --n_reps=1 --n_generations=2 --episode_steps=300 \
  --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 --lr=0.001 \
  --init_from=$INIT --lr_schedule=cosine --lr_total_gens=2 \
  --hebb_gate=1.0 --hebb_gate_att=45 --hebb_gate_mode=max --tag=smoke9

echo "=== 2. tiny replay ==="
run /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
  --results_dir results/${SLURM_JOB_ID}_smoke9 --which theta --num_envs 4 --n_episodes 1 --episode_steps 300 --lr 0.001

echo "=== 3. tiny hover test ==="
run /workspace/project/scripts/hebbian/replay_hebbian.py --headless \
  --results_dir results/${SLURM_JOB_ID}_smoke9 --zero_action --num_envs 4 --n_episodes 1 --episode_steps 300