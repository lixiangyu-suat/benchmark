#!/bin/bash

(return 0 2>/dev/null) && _SOURCED=1 || _SOURCED=0

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)"
[ -n "$_SDIR" ] && cd "$_SDIR/.."

MODEL_NAME="$1"
shift 2>/dev/null || true

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: source ${BASH_SOURCE[0]} <checkpoint_stem> [--save_viz]"
    [ "$_SOURCED" -eq 1 ] && return 1 || exit 1
fi

python src/evaluate.py \
    --model "$MODEL_NAME" \
    --cfg configs/config.yaml \
    "$@"