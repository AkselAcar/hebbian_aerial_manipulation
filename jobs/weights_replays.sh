#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH -J weights_replays
#SBATCH -o logs/%x_%A_%a.out

# Weight recording for plot_weights.py: replays the final controller (theta) of each run in RUNS with
# --record_weights (32 swarms x 1 episode, seed 0 = the same flights for every run). Does not train anything.
# Output: results/<run>/replay_theta_<stamp>/weights_ep0.npz
# Then (plain python3):  python3 plot_weights.py --runs <the same 4 folders>

RUNS=(results/3240667_A1_cos
      results/3240685_A1_cos_decay0.2
      results/3240622_A1_cos_gate1.0
      results/3247651_B1_gmean45
      results/3246978_A2_orient)
i=$SLURM_ARRAY_TASK_ID
RUN=${RUNS[$i]}
[ -n "$RUN" ] && [ -f "$RUN/theta.npy" ] || { echo "task $i: no run (found ${#RUNS[@]}: ${RUNS[*]})"; exit 1; }
echo "task $i: $RUN"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_weights$i
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
      --results_dir $RUN --which theta --num_envs 32 --n_episodes 1 --lr 0.001 --record_weights
  "