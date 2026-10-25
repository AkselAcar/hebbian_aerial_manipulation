#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH -J sweep2_oaies_adam_w_abcd
#SBATCH -o logs/%x_%A_%a.out

# OpenAI-ES sweep 2, around the setting that worked in sweep 1
# (pop 512, Adam lr 0.01, sigma 0.1 -> 70% full episodes after ~700 gens).
#   sigma {0.1, 0.05} x Adam lr {0.01, 0.02, 0.03} = 6 runs, one GPU each.
# All keep lr < sigma (Fuda's rule). Task 0 repeats sweep 1's best setting,
# to see how much two runs with the same settings differ.
# Fixed: pop 512 x 2 reps (1024 envs), genome w_abcd (5695), Hebbian lr 0.001,
# weight decay 0.005, es_optimizer adam. 48 h (sweep 1 took off after ~450 gens).
#   %A = array job id, %a = task number (0..5)
# Logs:  logs/sweep2_oaies_adam_w_abcd_<A>_<task>.out
# CSVs:  logs/generation_log_<jobid>_s<sigma>_eslr<lr>.csv

SIGMAS=(0.1  0.1  0.1  0.05 0.05 0.05)
ES_LRS=(0.01 0.02 0.03 0.01 0.02 0.03)

i=$SLURM_ARRAY_TASK_ID
SIGMA=${SIGMAS[$i]}
ES_LR=${ES_LRS[$i]}
TAG="s${SIGMA}_eslr${ES_LR}"
echo "array task $i: sigma=$SIGMA es_lr=$ES_LR tag=$TAG"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless \
      --pop_size=512 --n_reps=2 --n_generations=10000 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=$SIGMA --es_lr=$ES_LR \
      --weight_decay=0.005 --lr=0.001 --tag=$TAG
  "