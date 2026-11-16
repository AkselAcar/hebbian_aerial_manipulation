#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-7
#SBATCH -J review_replays
#SBATCH -o logs/%x_%A_%a.out

# Replays the final controller (theta) of each run in RUNS on the NORMAL task (normal ranges and
# reward weights, whatever the run trained on): 64 flights each (32 swarms x 2 episodes), no video.
# Fixed seed (replay default --seed 0): every run flies exactly the same 64 start/goal pairs.
# The plasticity brakes are read from each run's train_args.json.
# Output: results/<run>/replay_theta_<stamp>/
# Then (plain python3): diagnose_replay.py and plot_freeze.py on those folders (commands in the chat).

# Edit this list to replay other runs (task i replays RUNS[i]; set --array to 0-(number of runs - 1)).
RUNS=(results/3246978_A2_orient
      results/3247651_B1_gmean45
      results/3247895_B1_gmax45)
i=$SLURM_ARRAY_TASK_ID
RUN=${RUNS[$i]}
[ -n "$RUN" ] && [ -f "$RUN/theta.npy" ] || { echo "task $i: no run (found ${#RUNS[@]}: ${RUNS[*]})"; exit 1; }
echo "task $i: $RUN"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_review$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/replay_hebbian.py --headless \
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 2 --lr 0.001
  "