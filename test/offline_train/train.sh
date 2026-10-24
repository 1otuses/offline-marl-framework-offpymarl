#!/usr/bin/env bash
set -euo pipefail

# bash test/offline_train/train.sh bc 3m expert 1 0 [--t_max=5000 ...]
if (( $# < 5 )); then
    echo "Usage: $0 ALGORITHM MAP QUALITY SEED GPU [--key=value ...]" >&2
    exit 2
fi
algorithm=${1#offline/}
map=$2
quality=$3
seed=$4
gpu=$5
shift 5
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python -u src/main.py \
    --offline \
    --config="offline/$algorithm" \
    --env-config=sc2_offline \
    --map_name="$map" \
    --offline_data_quality="$quality" \
    --seed="$seed" \
    --g="$gpu" \
    --t_max=200000 \
    --test_interval=10000 \
    --log_interval=1000 \
    --runner_log_interval=250 \
    --learner_log_interval=1000 \
    --save_model_interval=10000 \
    "$@"
