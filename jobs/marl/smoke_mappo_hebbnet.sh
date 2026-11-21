#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH -J smoke_mappo_hebbnet
#SBATCH -o logs/%x_%j.out

# Quick check of the whole MAPPO -> Hebbian chain (~15-20 min, 3 Isaac start-ups), on the OmniNxt:
#   1. MAPPO with the Hebbian network, 64 envs, 256 steps (2 updates), one checkpoint
#   2. export of that checkpoint to a Hebbian run folder (results/mappo_smoke_<job>)
#   3. replay of the exported policy with replay_hebbian.py (the static network)
#   4. 2 tiny generations of train_hebbian.py --from_mappo (evolve A, B, C, D only)
# Look in the .out for:
#   "[mappo_hebbnet] policy parameters: {'log_std_parameter': (5,), 'net_container.0.weight': (15, 45), 'net_container.2.weight': (5, 15)}"
#   "network 45 -> 15 -> 5: 750 weights" (export), "start weights fixed from results/mappo_smoke_..." and
#   "genome 3000" (train_hebbian), the four "=== step N done" lines, and no Traceback.

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SKRL=$HOME/lis_semester_project/skrl
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_smoke_mappo
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1
grep -q shared_across_agents $SKRL/skrl/multi_agents/torch/mappo/mappo_cfg.py \
  || { echo "Jack's skrl fork in $SKRL has no parameter sharing: run 'cd $SKRL && git pull' first"; exit 1; }
EXP=results/mappo_smoke_$SLURM_JOB_ID

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B "$SKRL":/workspace/algos/skrl \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    set -e
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:/workspace/algos/skrl:\$PYTHONPATH
    PY=/workspace/isaaclab/_isaac_sim/python.sh
    cd /workspace/project
    \$PY scripts/skrl/train_mappo_hebbnet.py --headless --num_envs=64 --drone=omninxt \
      --timesteps=256 --checkpoint_interval=256 --tag=smoke
    echo '=== step 1 done'
    RUN=\$(ls -d logs/skrl/mappo_hebbnet/*_${SLURM_JOB_ID}_smoke | head -1)
    \$PY scripts/skrl/export_mappo_to_hebbian.py --checkpoint \$RUN/checkpoints/agent_256.pt --out $EXP
    echo '=== step 2 done'
    \$PY scripts/hebbian/replay_hebbian.py --headless --results_dir $EXP --num_envs=16 --n_episodes=1 --episode_steps=200
    echo '=== step 3 done'
    \$PY scripts/hebbian/train_hebbian.py --headless --from_mappo $EXP \
      --pop_size=8 --n_reps=1 --n_generations=2 --episode_steps=300 \
      --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 --weight_decay=0.005 --lr=0.001 \
      --ori_weight_start=6 --calm_weight=0 --hebb_decay=0.2 --tag=smoke_from_mappo
    echo '=== step 4 done'
  "