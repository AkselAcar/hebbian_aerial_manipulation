#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH -J oes_stepC2
#SBATCH -o logs/%x_%A_%a.out

# Step C2 (2026-11-17): FROM SCRATCH, old drone (Falcon), same base as C1 (reward = position + orientation,
# the 4 regularization terms at 0, collision termination off for 300 gens, cosine lr), PLUS the Hebbian weight
# decay 0.2 /s in every task (plot_weights.py: it stabilises the weights at no cost).
#   task 0 posonly:   orientation weight 0 -> how good can position get alone? (max fitness 30 per drone)
#   task 1 orifirst:  orientation first through the weights: orientation 3.0 -> 1.5 and position 0.5 -> 1.5
#                     over the first 500 gens
#   task 2 base:      nothing else (= C1 base + decay: does the decay help or hurt from scratch?)
#   task 3 orient_gate: orientation curriculum (as C1) + neuromodulation on position and orientation (mean)
# Needs train_hebbian v8.4 and the env with the drone option. Logs: logs/oes_stepC2_<A>_<task>.out
# Every "full flights" line also shows the rotor thrust saturation.

COMMON="--calm_weight=0 --collision_off_gens=300 --collision_ramp_gens=200 --hebb_decay=0.2"
CURR="--curr_yaw_gens=600 --curr_tilt_gens=200"
NAMES=(posonly orifirst base orient_gate)
EXTRA=("$COMMON --ori_weight_start=0"
       "$COMMON --ori_weight_start=3.0 --ori_weight_end=1.5 --ori_weight_gens=500 --pos_weight_start=0.5 --pos_weight_end=1.5 --pos_weight_gens=500"
       "$COMMON"
       "$COMMON $CURR --hebb_gate=1.0 --hebb_gate_att=45 --hebb_gate_mode=mean")
i=$SLURM_ARRAY_TASK_ID
TAG="C2_${NAMES[$i]}"
echo "task $i: $TAG ${EXTRA[$i]}"

REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_C2_$i
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
      --weight_decay=0.005 --lr=0.001 \
      --lr_schedule=cosine --lr_total_gens=1000 --lr_min_frac=0.1 ${EXTRA[$i]} --tag=$TAG
  "