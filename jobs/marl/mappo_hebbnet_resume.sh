#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH -J mappo_resume
#SBATCH -o logs/%x_%A_%a.out

# Continues the 4 MAPPO runs of jobs/marl/mappo_hebbnet.sh from their last checkpoint (they reached ~425k
# of 1.5M steps in 24 h). Task i takes the newest run folder ending in NAMES[i] (the first try or an
# earlier resume) and its highest agent_<step>.pt; policy, critic, optimizers and input statistics carry over.
# New log folder <date>_<job>_<name>_resumed; step counts restart from 0 there.
# 800k more steps ~ 44 h at the speed of the first try.

NAMES=(falcon_lr5e-5 falcon_lr3e-4 omninxt_lr5e-5 omninxt_lr3e-4)
EXTRA=("--drone=falcon --policy_lr=5e-5"
       "--drone=falcon --policy_lr=3e-4"
       "--drone=omninxt --policy_lr=5e-5"
       "--drone=omninxt --policy_lr=3e-4")
i=$SLURM_ARRAY_TASK_ID
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
cd $REPO
RUN=$(ls -td logs/skrl/mappo_hebbnet/*_${NAMES[$i]} logs/skrl/mappo_hebbnet/*_${NAMES[$i]}_resumed 2>/dev/null | head -1)
CKPT=$(ls $RUN/checkpoints/agent_*.pt 2>/dev/null | sed 's/.*agent_\([0-9]*\)\.pt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
[ -f "$CKPT" ] || { echo "task $i: no checkpoint found for ${NAMES[$i]} (run folder: $RUN)"; exit 1; }
echo "task $i: ${NAMES[$i]} resumes from $CKPT"

SKRL=$HOME/lis_semester_project/skrl
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
KIT=/scratch/izar/$USER/kit_mappo_$i
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B "$SKRL":/workspace/algos/skrl \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    export PYTHONPATH=/workspace/project/exts/MARL_mav_carry_ext:/workspace/algos/skrl:\$PYTHONPATH
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train_mappo_hebbnet.py --headless --num_envs=4096 \
      --ori_weight=6 --pos_weight=1.5 --calm_weight=0 ${EXTRA[$i]} --timesteps=800000 \
      --checkpoint $CKPT --tag=${NAMES[$i]}_resumed
  "