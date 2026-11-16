#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH -J oes_stepB1
#SBATCH -o logs/%x_%A_%a.out

# Step B1 (2026-11): neuromodulation on the orientation too (Fuda). Warm start from the winner of A1,
# the position-gated run (A1_cos_gate1.0), same settings otherwise (cosine lr 0.01 -> 0.001, 1000 gens).
# The Hebbian update of each drone is scaled by m = min(1, e):
#   task 0 gmax45:  e = max(pos error / 1 m, attitude error / 45 deg)   adapt while EITHER error is large
#   task 1 gmean45: e = mean(pos error / 1 m, attitude error / 45 deg)  softer: both errors count half
# Logs: logs/oes_stepB1_<A>_<task>.out   CSVs: logs/generation_log_<jobid>_B1_<name>.csv (with the
# per-generation position and attitude errors). Compare with A1_cos_gate1.0 by replays.

NAMES=(gmax45 gmean45)
EXTRA=("--hebb_gate=1.0 --hebb_gate_att=45 --hebb_gate_mode=max"
       "--hebb_gate=1.0 --hebb_gate_att=45 --hebb_gate_mode=mean")
i=$SLURM_ARRAY_TASK_ID
TAG="B1_${NAMES[$i]}"
INIT=results/3240622_A1_cos_gate1.0/theta.npy
[ -f "$INIT" ] || { echo "missing $INIT"; exit 1; }
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_B1_$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/hebbian/train_hebbian.py --headless \
      --pop_size=512 --n_reps=2 --n_generations=1000 --episode_steps=2000 \
      --genome=w_abcd --optimizer=openai_es --es_optimizer=adam --sigma0=0.05 --es_lr=0.01 \
      --weight_decay=0.005 --lr=0.001 --init_from=/workspace/project/$INIT \
      --lr_schedule=cosine --lr_total_gens=1000 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "