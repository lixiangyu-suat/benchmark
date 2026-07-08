#!/bin/bash

(return 0 2>/dev/null) && _SOURCED=1 || _SOURCED=0

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)"
[ -n "$_SDIR" ] && cd "$_SDIR/.."

TEACHER="$1"
STUDENT="$2"
shift 2

if [ -z "$TEACHER" ] || [ -z "$STUDENT" ]; then
    echo "Usage: source ${BASH_SOURCE[0]} <teacher_ckpt> <student_arch_or_ckpt>"
    echo
    echo "  <student_arch_or_ckpt> semantics:"
    echo "    - architecture name (e.g. Mobile_U_ViT)  ->  new student training from scratch"
    echo "    - checkpoint stem   (e.g. Mobile_U_ViT_model_2026-07-04_23_17_55)  ->  resume student"
    [ "$_SOURCED" -eq 1 ] && return 1 || exit 1
fi

# -- Auto-detect: student checkpoint stem contains "_model_" ---------
if [[ "$STUDENT" == *_model_* ]]; then
    python src/distill.py \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --cfg configs/config.yaml \
        --resume \
        "$@"
else
    python src/distill.py \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --cfg configs/config.yaml \
        "$@"
fi