#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH -J sweep_oaies_adam_w_abcd
#SBATCH -o logs/%x_%A_%a.out

# OpenAI-ES sweep following Fuda's advice (pop 512-2048, sigma 0.1, Adam,
# lr between 0.5 and 0.0001 depending on pop and sigma):
#   population {512, 1024} x Adam lr {0.01, 0.003, 0.001} = 6 runs, one GPU each.
# With Adam every parameter moves ~lr per generation, so lr is kept well below
# sigma (0.1): 0.01 = sigma/10 down to 0.001 = sigma/100.
# Fixed: genome w_abcd (5695), sigma 0.1, 2 reps per genome, Hebbian lr 0.001,
# weight decay 0.005, es_optimizer adam.
# Envs per run = pop x reps = 1024 or 2048.
#   %A = array job id (same for all runs), %a = task number (0..5)
# Logs:  logs/sweep_oaies_adam_w_abcd_<A>_<task>.out
# CSVs:  logs/generation_log_<jobid>_p<pop>_eslr<lr>.csv

POPS=(  512  512  512  1024 1024 1024)
ES_LRS=(0.01 0.003 0.001 0.01 0.003 0.001)

i=$SLURM_ARRAY_TASK_ID
POP=${POPS[$i]}
ES_LR=${ES_LRS[$i]}
TAG="p${POP}_eslr${ES_LR}"
echo "array task $i: pop=$POP es_lr=$ES_LR tag=$TAG"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif

apptainer exec --nv \
  -B "$REPO":/workspace/project \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/project/scripts/hebbian/train_hebbian.py --headless \
      --pop_size=$POP --n_reps=2 --n_generations=5000 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.1 --es_lr=$ES_LR \
      --weight_decay=0.005 --lr=0.001 --tag=$TAG
  "