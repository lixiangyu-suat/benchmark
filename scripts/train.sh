#!/bin/bash

(return 0 2>/dev/null) && _SOURCED=1 || _SOURCED=0

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)"
[ -n "$_SDIR" ] && cd "$_SDIR/.."

MODEL="$1"
shift 2>/dev/null || true

if [ -z "$MODEL" ]; then
    echo "Usage: source ${BASH_SOURCE[0]} <model_name>"
    echo
    echo "  <model_name> can be:"
    echo "    - an architecture name (e.g. U_Net, UNetplus)   ->  new training"
    echo "    - a checkpoint stem                              ->  resume from checkpoint"
    echo "        new: 20260708_1624_U_Net"
    echo "        old: UNetplus_model_2026-07-04_23_17_55"
    [ "$_SOURCED" -eq 1 ] && return 1 || exit 1
fi

# -- Auto-detect: checkpoint stem contains "_model_" -----------------
# New format: "20260708_1624_U_Net"
if [[ "$MODEL" =~ ^[0-9]{8}_[0-9]{4}_ ]]; then
    ARCH="${MODEL#*_*_}"
    python src/train.py \
        --model "$ARCH" \
        --ckpt "$MODEL" \
        --cfg configs/config.yaml \
        "$@"
# Old format: "UNetplus_model_2026-07-04_23_17_55"
elif [[ "$MODEL" == *_model_* ]]; then
    ARCH="${MODEL%%_model_*}"
    python src/train.py \
        --model "$ARCH" \
        --ckpt "$MODEL" \
        --cfg configs/config.yaml \
        "$@"
else
    python src/train.py \
        --model "$MODEL" \
        --cfg configs/config.yaml \
        "$@"
fi
