#!/usr/bin/env bash
set -euo pipefail

count="${1:-1}"
mode="${2:-state}"
environment="${3:-environment_0}"
checkpoint="${4:-models/ViTLSTM_model.pth}"

if [[ ! "$environment" =~ ^environment_[0-9]+$ ]]; then
    echo "invalid tree environment: $environment" >&2
    exit 2
fi

scene_dir="$FLIGHTMARE_PATH/flightpy/configs/vision/trees/$environment"
if [[ ! -d "$scene_dir" ]]; then
    echo "tree environment does not exist: $scene_dir" >&2
    exit 2
fi

base_config="$FLIGHTMARE_PATH/flightpy/configs/vision/config.yaml"
run_config="/tmp/vitfly-${environment}.yaml"
sed -E \
    "s|^([[:space:]]*env_folder:).*|\1 \"$environment\"|" \
    "$base_config" > "$run_config"

exec bash launch_evaluation.bash "$count" "$mode" "$run_config" "$checkpoint"
