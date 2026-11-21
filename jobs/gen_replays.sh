#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-6
#SBATCH -J gen_replays
#SBATCH -o logs/%x_%A_%a.out

# Replays one run at several generations on the normal task, same seed (same 64 flights) for each,
# to see how the flights change along training (plot_generations.py).
#   sbatch jobs/gen_replays.sh results/<run> theta "0 25 50 100 200 400 699"
#       theta: the search centre, saved every 25 gens in theta_hist/ (runs started after Nov 21)
#   sbatch jobs/gen_replays.sh results/<run> best "0 25 50 100 200 400 699"
#       best: the best sample so far (x_best.npy), for the older runs (C1, D1, E1 if started before the update)
# Set --array to 0-(number of generations - 1).
# Then: python3 plot_generations.py --replays results/<run>/replay_*gen* --out gens_<run>

RUN=$1; WHICH=${2:-theta}; GENS=(${3:-0 25 50 100 200 400 699})
i=$SLURM_ARRAY_TASK_ID
G=${GENS[$i]}
[ -n "$G" ] && [ -d "$RUN" ] || { echo "task $i: nothing to do (gens: ${GENS[*]})"; exit 0; }
echo "task $i: $RUN $WHICH gen $G"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_gen$i
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
      --results_dir $RUN --which $WHICH --gen $G --num_envs 32 --n_episodes 2 --lr 0.001
  "