#!/usr/bin/env bash
# 添加实例坐标差值损失
set -x

EXP_DIR=/irsa/build_code/UniPoly/weights/SN3
PY_ARGS=${@:1}

python -u /irsa/build_code/UniPoly/train/main_spacenet3.py \
    --output_dir ${EXP_DIR} \
    --num_workers 8 \
    --with_box_refine \
    --two_stage \
    ${PY_ARGS}
