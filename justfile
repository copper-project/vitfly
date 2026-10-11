set shell := ["bash", "-euc"]

image := "vitfly-noetic:local"
repo := justfile_directory()
build_volume := "vitfly-noetic-build"
devel_volume := "vitfly-noetic-devel"
logs_volume := "vitfly-noetic-logs"

default:
    @just --list

# Build the Ubuntu 20.04 / ROS 1 Noetic environment.
noetic-image:
    docker build \
        --build-arg USER_ID="$(id -u)" \
        --build-arg GROUP_ID="$(id -g)" \
        --tag {{image}} \
        --file {{repo}}/docker/noetic/Dockerfile \
        {{repo}}/docker/noetic

# Compile the mounted ViTFly checkout in persistent catkin volumes.
noetic-build:
    docker run --rm \
        --volume {{repo}}:/ws/src/vitfly \
        --volume {{build_volume}}:/ws/build \
        --volume {{devel_volume}}:/ws/devel \
        --volume {{logs_volume}}:/ws/logs \
        {{image}} \
        vitfly-catkin-build

# Verify ROS messages, Python dependencies, and the generated simulator node.
noetic-check:
    docker run --rm \
        --volume {{repo}}:/ws/src/vitfly \
        --volume {{devel_volume}}:/ws/devel \
        {{image}} \
        bash -lc 'source /ws/devel/setup.bash && python3 -c "import rospy, cv_bridge, torch, torchvision, scipy, pandas, uniplot, matplotlib.pyplot; from scipy.spatial.transform import Rotation; from dodgeros_msgs.msg import QuadState; from envsim_msgs.msg import ObstacleArray; print(\"ViTFly Noetic environment OK\")" && test -x /ws/devel/lib/envsim/visionsim_node'

# Open an interactive shell in the built Noetic workspace.
noetic-shell:
    docker run --rm --interactive --tty \
        --network host \
        --device nvidia.com/gpu=all \
        --env DISPLAY="${DISPLAY}" \
        --env XAUTHORITY=/tmp/host.xauthority \
        --env VITFLY_RVIZ=False \
        --volume "${XAUTHORITY}:/tmp/host.xauthority:ro" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume {{repo}}:/ws/src/vitfly \
        --volume {{build_volume}}:/ws/build \
        --volume {{devel_volume}}:/ws/devel \
        --volume {{logs_volume}}:/ws/logs \
        {{image}} \
        bash -lc 'source /ws/devel/setup.bash && exec bash'

# Generate labeled trajectories with the original privileged expert.
noetic-rollout count="1" mode="state" environment="environment_0" checkpoint="models/ViTLSTM_wall_safe.pth":
    docker run --rm --init \
        --network host \
        --device nvidia.com/gpu=all \
        --env DISPLAY="${DISPLAY}" \
        --env XAUTHORITY=/tmp/host.xauthority \
        --env VITFLY_RVIZ=False \
        --volume "${XAUTHORITY}:/tmp/host.xauthority:ro" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume {{repo}}:/ws/src/vitfly \
        --volume {{build_volume}}:/ws/build \
        --volume {{devel_volume}}:/ws/devel \
        --volume {{logs_volume}}:/ws/logs \
        {{image}} \
        bash -lc 'source /ws/devel/setup.bash && exec bash docker/noetic/rollout.bash {{count}} {{mode}} {{environment}} {{checkpoint}}'

# Collect validated deterministic-expert trajectories across distinct tree scenes.
noetic-collect environments="141,101,169,272,288,400,343,12,207,237,135,112":
    docker run --rm --init \
        --network host \
        --device nvidia.com/gpu=all \
        --env DISPLAY="${DISPLAY}" \
        --env XAUTHORITY=/tmp/host.xauthority \
        --env VITFLY_RVIZ=False \
        --volume "${XAUTHORITY}:/tmp/host.xauthority:ro" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume {{repo}}:/ws/src/vitfly \
        --volume {{build_volume}}:/ws/build \
        --volume {{devel_volume}}:/ws/devel \
        --volume {{logs_volume}}:/ws/logs \
        {{image}} \
        bash -lc 'source /ws/devel/setup.bash && exec bash docker/noetic/collect.bash "{{environments}}"'

# Fine-tune the published ViTLSTM checkpoint on the original + corrected tree mix.
finetune-tree epochs="12":
    python training/train.py \
        --config training/config/finetune_tree_dead_on.txt \
        --N_eps {{epochs}}

# Rebuild the four-sided planar wall safety dataset deterministically.
generate-pure-walls:
    python training/generate_pure_walls.py

check-pure-walls checkpoint="models/ViTLSTM_model.pth":
    python training/evaluate_pure_walls.py {{checkpoint}}
