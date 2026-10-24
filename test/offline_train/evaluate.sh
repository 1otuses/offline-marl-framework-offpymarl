#!/usr/bin/env bash
set -euo pipefail

# checkpoint_root 指向包含数字 step 子目录的 models 目录；step=0 选择最新。
if (( $# < 7 )); then
    echo "Usage: $0 ALGORITHM MAP QUALITY CHECKPOINT_ROOT STEP EVAL_SEED GPU [--key=value ...]" >&2
    exit 2
fi
algorithm=${1#offline/}
map=$2
quality=$3
checkpoint_root=$4
step=$5
seed=$6
gpu=$7
shift 7
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ "$step" != 0 && ! -f "$checkpoint_root/$step/agent.th" ]]; then
    echo "Checkpoint does not exist: $checkpoint_root/$step/agent.th" >&2
    exit 2
fi

python -u src/main.py --offline \
    --config="offline/$algorithm" \
    --env-config=sc2_offline \
    --map_name="$map" \
    --offline_data_quality="$quality" \
    --checkpoint_path="$checkpoint_root" \
    --load_step="$step" \
    --evaluate=True \
    --runner=episode \
    --batch_size_run=1 \
    --test_nepisode=100 \
    --seed="$seed" \
    --g="$gpu" \
    "$@"
