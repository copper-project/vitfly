#!/usr/bin/env bash
set -euo pipefail

environment_csv="${1:?comma-separated environment list required}"
IFS=',' read -r -a environments <<< "$environment_csv"

mkdir -p training/datasets/tree_dead_on
mkdir -p /tmp/vitfly-rejected-rollouts

for environment in "${environments[@]}"; do
    if [[ "$environment" =~ ^[0-9]+$ ]]; then
        environment="environment_$environment"
    fi
    echo "[COLLECT] Starting $environment"
    run_log="/tmp/vitfly-${environment}.log"
    if ! bash docker/noetic/rollout.bash 1 state "$environment" > "$run_log" 2>&1; then
        echo "[COLLECT] Launcher failed for $environment" >&2
        tail -40 "$run_log" >&2
        continue
    fi

    trajectory="$(find envtest/ros/train_set -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "$trajectory" || ! -f "$trajectory/data.csv" ]]; then
        echo "[COLLECT] No trajectory produced for $environment" >&2
        tail -40 "$run_log" >&2
        continue
    fi

    frames="$(find "$trajectory" -maxdepth 1 -name '*.png' | wc -l)"
    rows="$(awk 'END {print NR-1}' "$trajectory/data.csv")"
    duplicate_timestamps="$(cut -d, -f2 "$trajectory/data.csv" | tail -n +2 | sort | uniq -d | wc -l)"
    collision_rows="$(awk -F, 'NR>1 && $21=="True" {n++} END {print n+0}' "$trajectory/data.csv")"

    if grep -q 'Success: true' envtest/ros/summary.yaml \
        && grep -q 'number_crashes: 0' envtest/ros/summary.yaml \
        && { [[ "$frames" -eq "$rows" ]] || [[ "$frames" -eq $((rows + 1)) ]]; } \
        && [[ "$duplicate_timestamps" -eq 0 ]] \
        && [[ "$collision_rows" -eq 0 ]]; then
        destination="training/datasets/tree_dead_on/${environment}_$(basename "$trajectory")"
        mv "$trajectory" "$destination"
        echo "[COLLECT] Accepted $environment: $rows aligned samples"
    else
        destination="/tmp/vitfly-rejected-rollouts/${environment}_$(basename "$trajectory")"
        mv "$trajectory" "$destination"
        echo "[COLLECT] Rejected $environment (frames=$frames rows=$rows duplicates=$duplicate_timestamps collisions=$collision_rows)" >&2
        tail -25 "$run_log" >&2
    fi
done
