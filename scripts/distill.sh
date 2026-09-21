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
    echo "    - checkpoint stem                         ->  resume student"
    echo "        new: 20260708_1624_Mobile_U_ViT"
    echo "        old: Mobile_U_ViT_model_2026-07-04_23_17_55"
    [ "$_SOURCED" -eq 1 ] && return 1 || exit 1
fi

# -- Auto-detect: student checkpoint stem contains "_model_" ---------
# New format: "20260708_1624_Mobile_U_ViT"
if [[ "$STUDENT" =~ ^[0-9]{8}_[0-9]{4}_ ]]; then
    CUDA_VISIBLE_DEVICES=0 python src/distill.py \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --cfg configs/config.yaml \
        --resume \
        "$@"
# Old format: "Mobile_U_ViT_model_2026-07-04_23_17_55"
elif [[ "$STUDENT" == *_model_* ]]; then
    CUDA_VISIBLE_DEVICES=0 python src/distill.py \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --cfg configs/config.yaml \
        --resume \
        "$@"
else
    CUDA_VISIBLE_DEVICES=0 python src/distill.py \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --cfg configs/config.yaml \
        "$@"
fi
