#!/bin/bash
# Runs Mixtral 1.2B model

unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
unset RANK
unset LOAL_RANK
unset MASTER_ADDR
unset WORLD_SIZE
unset MASTER_IP

export CUBLAS_WORKSPACE_CONFIG=:4096:8    # for deterministic



python3 benchmarks.py --config config.yaml

