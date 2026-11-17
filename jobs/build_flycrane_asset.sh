#!/bin/bash -l
#SBATCH -A master
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J build_flycrane
#SBATCH -o logs/%x_%j.out

# Builds the flycrane URDF for one drone model and converts it to USD (Isaac Lab 3.0 converter).
#   sbatch jobs/build_flycrane_asset.sh omninxt          # OmniNxt drones (controllers/drone_params.py)
#   sbatch jobs/build_flycrane_asset.sh falcon           # Jack's Falcons, names fixed only (pipeline check)
#   sbatch jobs/build_flycrane_asset.sh omninxt --no_meshes   # if the converter fails on basket.dae
# Output: exts/.../assets/data/AMR/flycrane_<drone>/flycrane_<drone>/flycrane_<drone>.usda
# Look in the .out for "[flycrane] mass per drone ..." and "USD OK".

DRONE=${1:-omninxt}
EXTRA=${2:-}
REPO=$HOME/lis_semester_project/MARL_cooperative_aerial_manipulation_ext
SIF=$HOME/containers/isaaclab_3.0.0-beta2.sif
AMR=$REPO/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR
KIT=/scratch/izar/$USER/kit_build_$DRONE
mkdir -p $KIT/cache $KIT/data $KIT/logs || exit 1
rm -rf "$AMR/flycrane_$DRONE"        # a previous conversion of the same drone

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONUNBUFFERED=1 \
  -B "$REPO":/workspace/project \
  -B $KIT/cache:/isaac-sim/kit/cache -B $KIT/data:/isaac-sim/kit/data -B $KIT/logs:/isaac-sim/kit/logs \
  "$SIF" \
  /bin/bash -c "
    cd /workspace/project
    /workspace/isaaclab/_isaac_sim/python.sh scripts/assets/build_flycrane_asset.py --drone $DRONE $EXTRA || exit 1
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/scripts/tools/convert_urdf.py --headless \
      exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR/flycrane_rod_data/flycrane_$DRONE.urdf \
      exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR/flycrane_$DRONE \
      --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none
  "

USDA=$AMR/flycrane_$DRONE/flycrane_$DRONE/flycrane_$DRONE.usda
if [ -f "$USDA" ]; then echo "USD OK: $USDA"; else echo "!!! no USD at $USDA"; find "$AMR/flycrane_$DRONE" -maxdepth 3 2>/dev/null; fi