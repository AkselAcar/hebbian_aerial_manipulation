#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH -J oes_mass_rand
#SBATCH -o logs/%x_%A_%a.out

# OpenAI-ES with a random payload mass: a new mass, uniform in 1.0-1.8 kg (Jack's old
# range), for every swarm at every generation. Same settings as the best run 3219114
# (sigma 0.05, Adam lr 0.01, pop 512, weights + ABCD, Hebbian lr 0.001), but 3 reps per
# genome instead of 2 so each genome is averaged over 3 masses (1536 envs).
#   task 0: warm start from the best run's theta  -> fast, shows if it adapts to mass
#   task 1: from scratch (zeros)                  -> fair comparison, slower
# Logs:  logs/oes_mass_rand_<A>_<task>.out
# CSVs:  logs/generation_log_<jobid>_mass1.0-1.8_<warm|scratch>.csv  (incl. payload_mass_min/mean/max)

INITS=("results/3219114_s0.05_eslr0.01_BEST/theta.npy" "")
NAMES=(warm scratch)

i=$SLURM_ARRAY_TASK_ID
INIT=${INITS[$i]}
TAG="mass1.0-1.8_${NAMES[$i]}"
INIT_ARG=""
if [ -n "$INIT" ]; then
  [ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }
  INIT_ARG="--init_from=/workspace/project/$INIT"
fi
echo "task $i: tag=$TAG init=${INIT:-zeros}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_train$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless \
      --pop_size=512 --n_reps=3 --n_generations=10000 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 --mass_range 1.0 1.8 $INIT_ARG --tag=$TAG
  "